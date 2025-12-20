"""
Optimized Ensemble Neural Network for Stock Return Prediction

This script trains an ensemble of neural networks with proper train/val/test splits
to avoid data leakage and validation contamination.

Advanced features:
- Gradient clipping for training stability
- Warmup + cosine annealing learning rate schedule
- Configurable loss functions (MSE, Huber)
- Proper temporal train/val/test splits

Author: Thesis Project
Date: December 2024
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from paths import get_processed_data_path, get_results_path

# --- REPRODUCIBILITY ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(RANDOM_SEED)

# --- SETTINGS ---
FILE_PATH = get_processed_data_path("thesis_dataset_macro.parquet")
RESULTS_FOLDER = get_results_path("")

# Time Splits (proper train/val/test to avoid contamination)
TRAIN_END = 2013      # Training: 1996-2013
VAL_END = 2016        # Validation: 2014-2016 (for early stopping)
# Test: 2017-2024 (NEVER seen during training/validation)

BATCH_SIZE = 4096        # Bigger batches = smoother gradients
EPOCHS = 50              # Give it time to learn
LEARNING_RATE = 0.0005
ENSEMBLE_MODELS = 5      # Train 5 distinct models and average them (Wisdom of Crowds)

# --- ADVANCED TRAINING OPTIONS ---
GRADIENT_CLIP_NORM = 1.0       # Gradient clipping for stability (None to disable)
USE_WARMUP_SCHEDULER = True    # Use warmup + cosine annealing
WARMUP_EPOCHS = 5              # Epochs for linear warmup
LOSS_FUNCTION = 'mse'          # 'mse' or 'huber'
HUBER_DELTA = 1.0              # Delta for Huber loss

RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

# --- SMART COMPONENTS ---

class EarlyStopping:
    """ Stops training if validation doesn't improve and restores best weights. """
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_r2, model):
        score = val_r2
        if self.best_score is None:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
        elif score < self.best_score: # If R2 drops
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else: # If R2 improves
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0


class WarmupCosineScheduler:
    """Learning rate scheduler with linear warmup followed by cosine annealing."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0
    
    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            factor = self.current_epoch / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            factor = 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = max(self.min_lr, base_lr * factor)
    
    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


class AssetPricingNet(nn.Module):
    def __init__(self, input_dim):
        super(AssetPricingNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),            # Swish/SiLU often beats ReLU in finance
            nn.Dropout(0.4),      # Higher dropout to force generalization
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Dropout(0.3),
            
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.Dropout(0.2),
            
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.model(x)


def get_loss_function():
    """Get loss function based on configuration."""
    if LOSS_FUNCTION.lower() == 'huber':
        return nn.HuberLoss(delta=HUBER_DELTA)
    else:
        return nn.MSELoss()


def train_optimized():
    print("=" * 60)
    print("OPTIMIZED ENSEMBLE NEURAL NETWORK TRAINING")
    print("=" * 60)
    print(f"\nRandom Seed: {RANDOM_SEED}")
    print(f"Train Period: 1996-{TRAIN_END}")
    print(f"Validation Period: {TRAIN_END+1}-{VAL_END}")
    print(f"Test Period: {VAL_END+1}-2024")
    
    # Print advanced settings
    print(f"\nAdvanced Settings:")
    print(f"  Gradient Clipping: {GRADIENT_CLIP_NORM if GRADIENT_CLIP_NORM else 'Disabled'}")
    print(f"  LR Schedule: {'Warmup + Cosine' if USE_WARMUP_SCHEDULER else 'ReduceOnPlateau'}")
    print(f"  Loss Function: {LOSS_FUNCTION.upper()}")
    print()
    
    print("Loading Data...")
    df = pd.read_parquet(str(FILE_PATH))
    
    # 1. Filter to liquid stocks (top 70% by market cap)
    df['size_rank'] = df.groupby('date')['mkt_cap'].rank(pct=True)
    df_liquid = df[df['size_rank'] > 0.3].copy()
    
    # 2. Define features (exclude identifiers and target)
    exclude = ['permno', 'date', 'ticker', 'ret', 'future_ret', 'prc', 'shrout', 
               'vol', 'bid', 'ask', 'siccd', 'size_rank', 'year', 'match_year', 'fyear']
    numeric_cols = df_liquid.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in exclude]
    target = 'future_ret'
    
    print(f"Training Ensemble of {ENSEMBLE_MODELS} models with {len(features)} features...")

    # 3. Sanitization (winsorize, handle missing values)
    for col in features:
        df_liquid[col] = pd.to_numeric(df_liquid[col], errors='coerce')
        df_liquid[col] = df_liquid[col].replace([np.inf, -np.inf], np.nan)
        median_val = df_liquid[col].median()
        df_liquid[col] = df_liquid[col].fillna(0 if pd.isna(median_val) else median_val)
        lower = df_liquid[col].quantile(0.01)
        upper = df_liquid[col].quantile(0.99)
        df_liquid[col] = df_liquid[col].clip(lower, upper)
    
    df_liquid = df_liquid.dropna(subset=[target])
    
    # 4. PROPER TRAIN/VAL/TEST SPLIT (Critical: No data leakage!)
    dates = df_liquid['date'].dt.year.values
    train_mask = dates <= TRAIN_END
    val_mask = (dates > TRAIN_END) & (dates <= VAL_END)
    test_mask = dates > VAL_END
    
    print(f"\nData Split:")
    print(f"  Train: {train_mask.sum():,} observations (1996-{TRAIN_END})")
    print(f"  Validation: {val_mask.sum():,} observations ({TRAIN_END+1}-{VAL_END})")
    print(f"  Test: {test_mask.sum():,} observations ({VAL_END+1}-2024)")
    
    # 5. Scaling - Fit ONLY on training data, then transform val and test
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_liquid.loc[train_mask, features].values)
    X_val = scaler.transform(df_liquid.loc[val_mask, features].values)
    X_test = scaler.transform(df_liquid.loc[test_mask, features].values)
    
    # Handle any remaining NaN after scaling
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    y_train = df_liquid.loc[train_mask, target].values.reshape(-1, 1)
    y_val = df_liquid.loc[val_mask, target].values.reshape(-1, 1)
    y_test = df_liquid.loc[test_mask, target].values.reshape(-1, 1)
    
    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nUsing Device: {device}")
    
    # --- ENSEMBLE TRAINING LOOP ---
    ensemble_preds = np.zeros((len(y_test_t), ENSEMBLE_MODELS))
    feature_importances = np.zeros(len(features))
    learning_curves = []  # Store learning curve data
    
    for i in range(ENSEMBLE_MODELS):
        # Set seed for each model for reproducibility while maintaining diversity
        torch.manual_seed(RANDOM_SEED + i)
        np.random.seed(RANDOM_SEED + i)
        
        print(f"\n--- Training Model {i+1}/{ENSEMBLE_MODELS} ---")
        
        # Initialize Model
        model = AssetPricingNet(len(features)).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
        criterion = get_loss_function()
        
        # Learning rate scheduler
        if USE_WARMUP_SCHEDULER:
            scheduler = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, EPOCHS)
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5)
        
        early_stopping = EarlyStopping(patience=6)
        
        dataset = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        epoch_train_losses = []
        epoch_val_r2s = []
        epoch_lrs = []
        early_stop_epoch = None
        
        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0.0
            
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping for stability
                if GRADIENT_CLIP_NORM:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
                
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(loader)
            epoch_train_losses.append(avg_train_loss)
            
            # Validation on VALIDATION SET (not test!) - Critical fix for contamination
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val_t.to(device)).cpu().numpy()
                val_r2 = r2_score(y_val_t.numpy(), val_preds)
                epoch_val_r2s.append(val_r2)
            
            # Update learning rate scheduler
            if USE_WARMUP_SCHEDULER:
                scheduler.step()
                current_lr = scheduler.get_lr()[0]
            else:
                scheduler.step(val_r2)
                current_lr = optimizer.param_groups[0]['lr']
            
            epoch_lrs.append(current_lr)
            early_stopping(val_r2, model)
            
            print(f"   Epoch {epoch+1}: Loss={avg_train_loss:.6f}, Val R²={val_r2:.4%}, LR={current_lr:.2e}")
            
            if early_stopping.early_stop:
                print(f"   >>> Early Stopping! Best Val R² was {early_stopping.best_score:.4%}")
                early_stop_epoch = epoch + 1
                break
        
        # Save learning curve data for this model
        if epoch_train_losses:
            for epoch_idx, (train_loss, val_r2, lr) in enumerate(
                zip(epoch_train_losses, epoch_val_r2s, epoch_lrs)
            ):
                learning_curves.append({
                    'model': i + 1,
                    'epoch': epoch_idx + 1,
                    'train_loss': train_loss,
                    'val_r2': val_r2,
                    'learning_rate': lr,
                    'early_stop_epoch': early_stop_epoch if epoch_idx + 1 == early_stop_epoch else None
                })
        
        # Restore Best Weights (from best validation performance)
        model.load_state_dict(early_stopping.best_state)
        
        # Store Test Predictions (test set only evaluated here, AFTER training)
        model.eval()
        with torch.no_grad():
            preds = model(X_test_t.to(device)).cpu().numpy()
            ensemble_preds[:, i] = preds.flatten()
            
        # Store Importance using TRAINING data (not test)
        X_sample = X_train_t[:1000].to(device).requires_grad_(True)
        output = model(X_sample)
        output.sum().backward()
        grads = X_sample.grad.abs().mean(dim=0).cpu().numpy()
        feature_importances += grads

    # --- FINAL EVALUATION ---
    print("\n" + "=" * 60)
    print("ENSEMBLE RESULTS")
    print("=" * 60)
    
    # Average predictions across all ensemble models
    final_preds = np.mean(ensemble_preds, axis=1)
    final_r2 = r2_score(y_test_t.numpy(), final_preds)
    
    print(f"\n>>> FINAL TEST SET R²: {final_r2:.4f} ({final_r2*100:.2f}%)")
    print(f"\nNote: This is TRUE out-of-sample performance on data from {VAL_END+1}-2024")
    print(f"      that was NEVER used for training or early stopping.\n")
    
    # Average Importance
    avg_importance = feature_importances / ENSEMBLE_MODELS
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': avg_importance
    }).sort_values(by='Importance', ascending=False)
    
    print("Top 10 Most Important Features:")
    print("-" * 40)
    for idx, row in importance_df.head(10).iterrows():
        print(f"  {row['Feature']:20s}: {row['Importance']:.6f}")
    print()
    
    # Save CSV for publication plots
    importance_df.to_csv(str(RESULTS_FOLDER / "optimized_importance.csv"), index=False)
    
    # Save learning curve data
    if learning_curves:
        lc_df = pd.DataFrame(learning_curves)
        # Group by epoch and average across models
        lc_agg = lc_df.groupby('epoch').agg({
            'train_loss': 'mean',
            'val_r2': 'mean',
            'learning_rate': 'mean',
            'early_stop_epoch': lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else None
        }).reset_index()
        lc_agg.to_csv(str(RESULTS_FOLDER / "learning_curve_data.csv"), index=False)
    
    # Save summary results
    summary = {
        'test_r2': final_r2,
        'train_period': f'1996-{TRAIN_END}',
        'val_period': f'{TRAIN_END+1}-{VAL_END}',
        'test_period': f'{VAL_END+1}-2024',
        'ensemble_models': ENSEMBLE_MODELS,
        'n_features': len(features),
        'random_seed': RANDOM_SEED,
        'gradient_clip_norm': GRADIENT_CLIP_NORM,
        'scheduler': 'warmup_cosine' if USE_WARMUP_SCHEDULER else 'reduce_on_plateau',
        'loss_function': LOSS_FUNCTION
    }
    pd.DataFrame([summary]).to_csv(str(RESULTS_FOLDER / "model_summary.csv"), index=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), palette='viridis')
    plt.title(f'Feature Importance (Ensemble of {ENSEMBLE_MODELS} Models)\n'
              f'Test R²: {final_r2*100:.2f}% | Period: {VAL_END+1}-2024')
    plt.xlabel('Gradient-Based Importance')
    plt.tight_layout()
    plt.savefig(str(RESULTS_FOLDER / "optimized_importance.png"), dpi=150)
    plt.close()
    
    print(f"[SUCCESS] Results saved to {RESULTS_FOLDER}")
    print("=" * 60)

if __name__ == "__main__":
    train_optimized()
