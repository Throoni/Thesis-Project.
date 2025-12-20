"""
Hyperparameter Tuning with Optuna

This script uses Optuna to find optimal hyperparameters for the neural network:
- Learning rate, batch size, hidden dimensions, dropout rates
- Tree-structured Parzen Estimator (TPE) sampler
- Median pruner for early stopping of bad trials
- Visualization of optimization history

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
import sys
from pathlib import Path
import json
import warnings

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from paths import get_processed_data_path, get_results_path
from models import set_seed, get_device, sanitize_features

# Check for Optuna
try:
    import optuna
    from optuna.visualization import plot_optimization_history, plot_param_importances
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("Warning: Optuna not installed. Run: pip install optuna")

# --- SETTINGS ---
RANDOM_SEED = 42
set_seed(RANDOM_SEED)

FILE_PATH = get_processed_data_path("thesis_dataset_macro.parquet")
RESULTS_FOLDER = get_results_path("")
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

# Time Splits
TRAIN_END = 2013
VAL_END = 2016

# Tuning Settings
N_TRIALS = 50           # Number of optimization trials
EPOCHS_PER_TRIAL = 20   # Epochs per trial (fewer for speed)
PATIENCE = 5            # Early stopping patience within trials


class FlexibleAssetPricingNet(nn.Module):
    """Neural network with configurable architecture for hyperparameter tuning."""
    
    def __init__(self, input_dim, hidden_dims, dropout_rates, activation='silu'):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim, dropout in zip(hidden_dims, dropout_rates):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            
            if activation == 'silu':
                layers.append(nn.SiLU())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


def prepare_data():
    """Load and prepare data for tuning."""
    print("Loading data...")
    df = pd.read_parquet(str(FILE_PATH))
    
    # Filter to liquid stocks
    df['size_rank'] = df.groupby('date')['mkt_cap'].rank(pct=True)
    df_liquid = df[df['size_rank'] > 0.3].copy()
    
    # Define features
    exclude = ['permno', 'date', 'ticker', 'ret', 'future_ret', 'prc', 'shrout', 
               'vol', 'bid', 'ask', 'siccd', 'size_rank', 'year', 'match_year', 'fyear']
    numeric_cols = df_liquid.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in exclude]
    target = 'future_ret'
    
    # Sanitize
    df_liquid = sanitize_features(df_liquid, features)
    df_liquid = df_liquid.dropna(subset=[target])
    
    # Time splits
    dates = df_liquid['date'].dt.year.values
    train_mask = dates <= TRAIN_END
    val_mask = (dates > TRAIN_END) & (dates <= VAL_END)
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_liquid.loc[train_mask, features].values)
    X_val = scaler.transform(df_liquid.loc[val_mask, features].values)
    
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)
    
    y_train = df_liquid.loc[train_mask, target].values
    y_val = df_liquid.loc[val_mask, target].values
    
    return X_train, y_train, X_val, y_val, len(features), scaler


def objective(trial, X_train, y_train, X_val, y_val, n_features, device):
    """Optuna objective function for hyperparameter optimization."""
    
    # Sample hyperparameters
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [1024, 2048, 4096, 8192])
    
    # Architecture
    n_layers = trial.suggest_int('n_layers', 2, 4)
    
    hidden_dims = []
    dropout_rates = []
    
    prev_dim = n_features
    for i in range(n_layers):
        # Each layer should be smaller or equal to the previous
        max_dim = min(prev_dim, 256)
        min_dim = 16
        if max_dim < min_dim:
            max_dim = min_dim
        
        hidden_dim = trial.suggest_int(f'hidden_dim_{i}', min_dim, max_dim, step=16)
        dropout = trial.suggest_float(f'dropout_{i}', 0.1, 0.5)
        
        hidden_dims.append(hidden_dim)
        dropout_rates.append(dropout)
        prev_dim = hidden_dim
    
    activation = trial.suggest_categorical('activation', ['silu', 'relu', 'gelu'])
    
    # Create model
    model = FlexibleAssetPricingNet(
        n_features, 
        hidden_dims, 
        dropout_rates, 
        activation
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    # Data loaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
    
    best_val_r2 = float('-inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS_PER_TRIAL):
        # Train
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Validate
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t).cpu().numpy()
            val_r2 = r2_score(y_val_t.numpy(), val_preds)
        
        # Report intermediate value for pruning
        trial.report(val_r2, epoch)
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        # Early stopping within trial
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break
    
    return best_val_r2


def run_hyperparameter_tuning():
    """Run the full hyperparameter tuning process."""
    if not HAS_OPTUNA:
        print("ERROR: Optuna required. Install with: pip install optuna")
        return None
    
    print("=" * 60)
    print("HYPERPARAMETER TUNING WITH OPTUNA")
    print("=" * 60)
    print(f"\nTrials: {N_TRIALS}")
    print(f"Epochs per trial: {EPOCHS_PER_TRIAL}")
    print(f"Random seed: {RANDOM_SEED}")
    
    # Prepare data
    X_train, y_train, X_val, y_val, n_features, scaler = prepare_data()
    print(f"\nData prepared:")
    print(f"  Train: {len(X_train):,}")
    print(f"  Val: {len(X_val):,}")
    print(f"  Features: {n_features}")
    
    device = get_device()
    print(f"  Device: {device}")
    
    # Create study
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name='asset_pricing_nn'
    )
    
    print("\n" + "-" * 60)
    print("Starting optimization...")
    print("-" * 60)
    
    # Suppress Optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, n_features, device),
        n_trials=N_TRIALS,
        show_progress_bar=True
    )
    
    # Results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    
    print(f"\nBest Trial: #{study.best_trial.number}")
    print(f"Best Val R²: {study.best_value:.4%}")
    
    print("\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results = {
        'best_trial_number': study.best_trial.number,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials),
        'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    }
    
    with open(str(RESULTS_FOLDER / "hyperparameter_tuning_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save all trials
    trials_df = study.trials_dataframe()
    trials_df.to_csv(str(RESULTS_FOLDER / "hyperparameter_trials.csv"), index=False)
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    # Optimization history
    try:
        fig = plot_optimization_history(study)
        fig.write_image(str(RESULTS_FOLDER / "optimization_history.png"))
        print("  Saved: optimization_history.png")
    except Exception as e:
        print(f"  Warning: Could not create optimization history plot: {e}")
        
        # Fallback matplotlib plot
        plt.figure(figsize=(10, 6))
        trials = study.trials
        values = [t.value for t in trials if t.value is not None]
        best_values = [max(values[:i+1]) for i in range(len(values))]
        plt.plot(values, 'o-', alpha=0.5, label='Trial Value')
        plt.plot(best_values, 'r-', linewidth=2, label='Best Value')
        plt.xlabel('Trial')
        plt.ylabel('Validation R²')
        plt.title('Hyperparameter Optimization History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(str(RESULTS_FOLDER / "optimization_history.png"), dpi=150)
        plt.close()
        print("  Saved: optimization_history.png (matplotlib fallback)")
    
    # Parameter importance
    try:
        fig = plot_param_importances(study)
        fig.write_image(str(RESULTS_FOLDER / "param_importance.png"))
        print("  Saved: param_importance.png")
    except Exception as e:
        print(f"  Warning: Could not create param importance plot: {e}")
    
    # Create best config file
    best_config = {
        'epochs': 50,
        'batch_size': study.best_params.get('batch_size', 4096),
        'learning_rate': study.best_params.get('learning_rate', 0.0005),
        'weight_decay': study.best_params.get('weight_decay', 1e-3),
        'n_layers': study.best_params.get('n_layers', 3),
        'hidden_dims': [
            study.best_params.get(f'hidden_dim_{i}', 64) 
            for i in range(study.best_params.get('n_layers', 3))
        ],
        'dropout_rates': [
            study.best_params.get(f'dropout_{i}', 0.3) 
            for i in range(study.best_params.get('n_layers', 3))
        ],
        'activation': study.best_params.get('activation', 'silu'),
        'best_val_r2': study.best_value
    }
    
    with open(str(RESULTS_FOLDER / "best_hyperparameters.json"), 'w') as f:
        json.dump(best_config, f, indent=2)
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Hyperparameter tuning complete!")
    print(f"Results saved to: {RESULTS_FOLDER}")
    print("=" * 60 + "\n")
    
    # Print configuration for use in training script
    print("To use best hyperparameters, update training config:")
    print("-" * 40)
    print(f"learning_rate = {best_config['learning_rate']:.6f}")
    print(f"batch_size = {best_config['batch_size']}")
    print(f"weight_decay = {best_config['weight_decay']:.6f}")
    print(f"hidden_dims = {tuple(best_config['hidden_dims'])}")
    print(f"dropout_rates = {tuple(best_config['dropout_rates'])}")
    print("-" * 40 + "\n")
    
    return study, results


if __name__ == "__main__":
    run_hyperparameter_tuning()

