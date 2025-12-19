"""
Permutation Feature Importance for Neural Network

More robust than gradient-based importance because it measures
the actual drop in performance when a feature is randomly shuffled.

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

# Try to import tqdm for progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("[INFO] tqdm not installed. Progress bar disabled.")

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from paths import get_processed_data_path, get_results_path

# --- SETTINGS ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

FILE_PATH = get_processed_data_path("thesis_dataset_macro.parquet")
RESULTS_FOLDER = get_results_path("")
N_PERMUTATIONS = 10  # Number of permutations per feature

# Time Splits
TRAIN_END = 2013
VAL_END = 2016


class AssetPricingNet(nn.Module):
    """Neural network architecture (must match training script)."""
    def __init__(self, input_dim):
        super(AssetPricingNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.4),
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


def permutation_importance(model, X, y, features, n_permutations=10, device='cpu'):
    """
    Calculate permutation importance for each feature.
    
    For each feature:
    1. Compute baseline R² with original data
    2. Shuffle the feature column
    3. Compute new R²
    4. Importance = baseline R² - shuffled R²
    
    Higher importance = bigger drop in performance when shuffled.
    """
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    
    # Baseline performance
    with torch.no_grad():
        baseline_preds = model(X_tensor).cpu().numpy().flatten()
    baseline_r2 = r2_score(y, baseline_preds)
    
    importance_scores = {}
    
    print(f"\nCalculating permutation importance for {len(features)} features...")
    print(f"Baseline R²: {baseline_r2:.4%}")
    print(f"Permutations per feature: {n_permutations}")
    print()
    
    # Use tqdm if available
    iterator = tqdm(enumerate(features), total=len(features), desc="Features") if HAS_TQDM else enumerate(features)
    
    for i, feature in iterator:
        r2_drops = []
        
        for _ in range(n_permutations):
            # Copy and shuffle one feature
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])
            
            # Predict with shuffled feature
            X_perm_tensor = torch.tensor(X_permuted, dtype=torch.float32).to(device)
            with torch.no_grad():
                perm_preds = model(X_perm_tensor).cpu().numpy().flatten()
            
            perm_r2 = r2_score(y, perm_preds)
            r2_drops.append(baseline_r2 - perm_r2)
        
        importance_scores[feature] = {
            'mean': np.mean(r2_drops),
            'std': np.std(r2_drops),
            'min': np.min(r2_drops),
            'max': np.max(r2_drops)
        }
        
        if not HAS_TQDM and (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(features)} features...")
    
    return importance_scores, baseline_r2


def main():
    print("=" * 60)
    print("PERMUTATION FEATURE IMPORTANCE")
    print("=" * 60)
    
    # Load and prepare data (same as training script)
    print("\nLoading data...")
    df = pd.read_parquet(str(FILE_PATH))
    
    df['size_rank'] = df.groupby('date')['mkt_cap'].rank(pct=True)
    df_liquid = df[df['size_rank'] > 0.3].copy()
    
    exclude = ['permno', 'date', 'ticker', 'ret', 'future_ret', 'prc', 'shrout', 
               'vol', 'bid', 'ask', 'siccd', 'size_rank', 'year', 'match_year', 'fyear']
    numeric_cols = df_liquid.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in exclude]
    target = 'future_ret'
    
    print(f"Features: {len(features)}")
    
    # Sanitization
    print("Sanitizing features...")
    for col in features:
        df_liquid[col] = pd.to_numeric(df_liquid[col], errors='coerce')
        df_liquid[col] = df_liquid[col].replace([np.inf, -np.inf], np.nan)
        median_val = df_liquid[col].median()
        df_liquid[col] = df_liquid[col].fillna(0 if pd.isna(median_val) else median_val)
        lower = df_liquid[col].quantile(0.01)
        upper = df_liquid[col].quantile(0.99)
        df_liquid[col] = df_liquid[col].clip(lower, upper)
    
    df_liquid = df_liquid.dropna(subset=[target])
    
    # Split
    dates = df_liquid['date'].dt.year.values
    train_mask = dates <= TRAIN_END
    test_mask = dates > VAL_END
    
    print(f"Train size: {train_mask.sum():,}")
    print(f"Test size:  {test_mask.sum():,}")
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_liquid.loc[train_mask, features].values)
    X_test = scaler.transform(df_liquid.loc[test_mask, features].values)
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    y_train = df_liquid.loc[train_mask, target].values.reshape(-1, 1)
    y_test = np.array(df_liquid.loc[test_mask, target].values).flatten()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Train a single model for importance calculation
    print("\nTraining model for importance calculation...")
    torch.manual_seed(RANDOM_SEED)
    model = AssetPricingNet(len(features)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)
    criterion = nn.MSELoss()
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=4096, shuffle=True)
    
    model.train()
    for epoch in range(30):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/30 complete")
    
    # Calculate permutation importance on TEST set
    importance_scores, baseline_r2 = permutation_importance(
        model, X_test, y_test, features, 
        n_permutations=N_PERMUTATIONS, device=device
    )
    
    # Create DataFrame
    importance_df = pd.DataFrame([
        {
            'Feature': feat,
            'Importance': scores['mean'],
            'Std': scores['std'],
            'Min': scores['min'],
            'Max': scores['max']
        }
        for feat, scores in importance_scores.items()
    ]).sort_values('Importance', ascending=False)
    
    # Display top 20
    print("\n" + "=" * 60)
    print("TOP 20 FEATURES BY PERMUTATION IMPORTANCE")
    print("=" * 60)
    print(f"\nBaseline Test R²: {baseline_r2:.4%}")
    print()
    print(f"{'Feature':<25} {'Importance':<12} {'Std':<10}")
    print("-" * 50)
    for _, row in importance_df.head(20).iterrows():
        print(f"{row['Feature']:<25} {row['Importance']:>10.6f}  ±{row['Std']:.6f}")
    
    # Save
    importance_df.to_csv(str(RESULTS_FOLDER / "permutation_importance.csv"), index=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    top_15 = importance_df.head(15)
    
    # Color by category
    macro_keywords = ['vix', 'inflation', 'cpi', 'term_spread', 'default_spread', 
                     'risk_free', 'sentiment']
    colors = ['#2E86AB' if any(kw in f.lower() for kw in macro_keywords) 
              else '#A23B72' for f in top_15['Feature']]
    
    bars = ax.barh(range(len(top_15)), top_15['Importance'] * 100, color=colors)
    ax.set_yticks(range(len(top_15)))
    ax.set_yticklabels(top_15['Feature'])
    ax.set_xlabel('Importance (% Drop in R² when permuted)', fontweight='bold')
    ax.set_title(f'Permutation Feature Importance\n(Baseline R² = {baseline_r2:.2%})', 
                 fontweight='bold')
    ax.invert_yaxis()
    
    # Add error bars
    ax.errorbar(top_15['Importance'] * 100, range(len(top_15)), 
                xerr=top_15['Std'] * 100, fmt='none', color='black', capsize=3)
    
    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='#2E86AB', label='Macro'),
        Patch(color='#A23B72', label='Stock-Level')
    ], loc='lower right')
    
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(RESULTS_FOLDER / "permutation_importance.png"), dpi=150)
    plt.close()
    
    print(f"\n[SUCCESS] Results saved to {RESULTS_FOLDER}")
    print("=" * 60)


if __name__ == "__main__":
    main()

