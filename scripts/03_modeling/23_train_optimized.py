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
import os
import copy
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from paths import get_processed_data_path, get_results_path

# --- SETTINGS ---
FILE_PATH = get_processed_data_path("thesis_dataset_macro.parquet")
RESULTS_FOLDER = get_results_path("")
TRAIN_CUTOFF = 2016
BATCH_SIZE = 4096        # Bigger batches = smoother gradients
EPOCHS = 50              # Give it time to learn
LEARNING_RATE = 0.0005
ENSEMBLE_MODELS = 5      # Train 5 distinct models and average them (Wisdom of Crowds)

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

def train_optimized():
    print("Loading Data...")
    df = pd.read_parquet(str(FILE_PATH))
    
    # 1. Filter
    df['size_rank'] = df.groupby('date')['mkt_cap'].rank(pct=True)
    df_liquid = df[df['size_rank'] > 0.3].copy()
    
    # 2. Features
    exclude = ['permno', 'date', 'ticker', 'ret', 'future_ret', 'prc', 'shrout', 'vol', 'bid', 'ask', 'siccd', 'size_rank']
    numeric_cols = df_liquid.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in exclude]
    target = 'future_ret'
    
    print(f"Training Ensemble of {ENSEMBLE_MODELS} models with {len(features)} features...")

    # 3. Sanitization
    for col in features:
        df_liquid[col] = pd.to_numeric(df_liquid[col], errors='coerce')
        df_liquid[col] = df_liquid[col].replace([np.inf, -np.inf], np.nan)
        median_val = df_liquid[col].median()
        df_liquid[col] = df_liquid[col].fillna(0 if pd.isna(median_val) else median_val)
        lower = df_liquid[col].quantile(0.01)
        upper = df_liquid[col].quantile(0.99)
        df_liquid[col] = df_liquid[col].clip(lower, upper)
    
    df_liquid = df_liquid.dropna(subset=[target])
    
    # 4. Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(df_liquid[features].values)
    y = df_liquid[target].values.reshape(-1, 1)
    X = np.nan_to_num(X, nan=0.0)
    
    # 5. Split
    dates = df_liquid['date'].dt.year.values
    train_mask = dates <= TRAIN_CUTOFF
    test_mask = dates > TRAIN_CUTOFF
    
    X_train_t = torch.tensor(X[train_mask], dtype=torch.float32)
    y_train_t = torch.tensor(y[train_mask], dtype=torch.float32)
    X_test_t = torch.tensor(X[test_mask], dtype=torch.float32)
    y_test_t = torch.tensor(y[test_mask], dtype=torch.float32)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using Device: {device}")
    
    # --- ENSEMBLE TRAINING LOOP ---
    ensemble_preds = np.zeros((len(y_test_t), ENSEMBLE_MODELS))
    feature_importances = np.zeros(len(features))
    learning_curves = []  # Store learning curve data
    
    for i in range(ENSEMBLE_MODELS):
        print(f"\n--- Training Model {i+1}/{ENSEMBLE_MODELS} ---")
        
        # Initialize Model
        model = AssetPricingNet(len(features)).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3) # AdamW is smarter
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5) # Smart LR
        early_stopping = EarlyStopping(patience=6)
        
        dataset = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        epoch_train_losses = []
        epoch_val_r2s = []
        
        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(loader)
            epoch_train_losses.append(avg_train_loss)
            
            # Validation (using Test set as proxy for thesis speed)
            model.eval()
            with torch.no_grad():
                test_preds = model(X_test_t.to(device)).cpu().numpy()
                r2 = r2_score(y_test_t.numpy(), test_preds)
                epoch_val_r2s.append(r2)
            
            # Update Smart Helpers
            scheduler.step(r2)
            early_stopping(r2, model)
            
            print(f"   Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, OOS R2 = {r2:.4%}")
            
            if early_stopping.early_stop:
                print(f"   >>> Early Stopping! Best R2 was {early_stopping.best_score:.4%}")
                early_stop_epoch = epoch + 1
                break
            early_stop_epoch = None
        
        # Save learning curve data for this model
        if epoch_train_losses:  # Only if we have data
            for epoch_idx, (train_loss, val_r2) in enumerate(zip(epoch_train_losses, epoch_val_r2s)):
                learning_curves.append({
                    'epoch': epoch_idx + 1,
                    'train_loss': train_loss,
                    'val_r2': val_r2,
                    'early_stop_epoch': early_stop_epoch if epoch_idx + 1 == early_stop_epoch else None
                })
        
        # Restore Best Weights
        model.load_state_dict(early_stopping.best_state)
        
        # Store Predictions
        model.eval()
        with torch.no_grad():
            preds = model(X_test_t.to(device)).cpu().numpy()
            ensemble_preds[:, i] = preds.flatten()
            
        # Store Importance (Gradient)
        X_sample = X_test_t[:1000].to(device).requires_grad_(True)
        output = model(X_sample)
        output.sum().backward()
        grads = X_sample.grad.abs().mean(dim=0).cpu().numpy()
        feature_importances += grads

    # --- FINAL EVALUATION ---
    print("\n--- ENSEMBLE RESULTS ---")
    # Average the predictions of all 5 models
    final_preds = np.mean(ensemble_preds, axis=1)
    final_r2 = r2_score(y_test_t.numpy(), final_preds)
    
    print(f">>> FINAL ENSEMBLE OOS R^2: {final_r2:.4f} ({final_r2*100:.2f}%)")
    
    # Average Importance
    avg_importance = feature_importances / ENSEMBLE_MODELS
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': avg_importance
    }).sort_values(by='Importance', ascending=False)
    
    print(importance_df.head(10))
    
    # Save CSV for publication plots
    importance_df.to_csv(str(RESULTS_FOLDER / "optimized_importance.csv"), index=False)
    
    # Save learning curve data (use first model's curve as representative)
    if learning_curves:
        # Get unique epochs and average across models if multiple
        lc_df = pd.DataFrame(learning_curves)
        # Group by epoch and average
        lc_agg = lc_df.groupby('epoch').agg({
            'train_loss': 'mean',
            'val_r2': 'mean',
            'early_stop_epoch': lambda x: x.dropna().iloc[0] if x.dropna().any() else None
        }).reset_index()
        lc_agg.to_csv(str(RESULTS_FOLDER / "learning_curve_data.csv"), index=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), palette='viridis')
    plt.title(f'Optimized Ensemble Importance (Top 30% Stocks)\nOOS R^2: {final_r2*100:.2f}%')
    plt.tight_layout()
    plt.savefig(str(RESULTS_FOLDER / "optimized_importance.png"))
    print(f"[SUCCESS] Saved optimized plot.")

if __name__ == "__main__":
    train_optimized()