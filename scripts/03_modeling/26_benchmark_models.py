"""
Benchmark Models Comparison

Compares neural network against:
1. Historical Mean (naive baseline)
2. Linear Regression (OLS)
3. Ridge Regression (regularized linear)
4. Random Forest
5. Gradient Boosting (XGBoost/LightGBM)

Author: Thesis Project
Date: December 2024
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import xgboost/lightgbm
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[INFO] XGBoost not installed. Skipping XGBoost benchmark.")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[INFO] LightGBM not installed. Skipping LightGBM benchmark.")

sys.path.append(str(Path(__file__).parent.parent / "utils"))
from paths import get_processed_data_path, get_results_path

# --- SETTINGS ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

FILE_PATH = get_processed_data_path("thesis_dataset_macro.parquet")
RESULTS_FOLDER = get_results_path("")
TRAIN_END = 2013
VAL_END = 2016


class AssetPricingNet(nn.Module):
    """Neural network for benchmarking."""
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.SiLU(), nn.Dropout(0.4),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(32, 16), nn.BatchNorm1d(16), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(16, 1)
        )
    def forward(self, x): 
        return self.model(x)


def train_neural_net(X_train, y_train, X_test, n_features, device, epochs=30):
    """Train neural network and return predictions."""
    torch.manual_seed(RANDOM_SEED)
    
    model = AssetPricingNet(n_features).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)
    criterion = nn.MSELoss()
    
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=4096, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32).to(device))
        return preds.cpu().numpy().flatten()


def main():
    print("=" * 60)
    print("BENCHMARK MODELS COMPARISON")
    print("=" * 60)
    
    # Load and prepare data
    print("\nLoading data...")
    df = pd.read_parquet(str(FILE_PATH))
    
    df['size_rank'] = df.groupby('date')['mkt_cap'].rank(pct=True)
    df_liquid = df[df['size_rank'] > 0.3].copy()
    
    exclude = ['permno', 'date', 'ticker', 'ret', 'future_ret', 'prc', 'shrout', 
               'vol', 'bid', 'ask', 'siccd', 'size_rank', 'year', 'match_year', 'fyear']
    numeric_cols = df_liquid.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in exclude]
    target = 'future_ret'
    
    # Sanitize
    print("Sanitizing features...")
    for col in features:
        df_liquid[col] = pd.to_numeric(df_liquid[col], errors='coerce')
        df_liquid[col] = df_liquid[col].replace([np.inf, -np.inf], np.nan)
        median_val = df_liquid[col].median()
        df_liquid[col] = df_liquid[col].fillna(0 if pd.isna(median_val) else median_val)
        lower, upper = df_liquid[col].quantile(0.01), df_liquid[col].quantile(0.99)
        df_liquid[col] = df_liquid[col].clip(lower, upper)
    df_liquid = df_liquid.dropna(subset=[target])
    
    dates = df_liquid['date'].dt.year.values
    train_mask = dates <= VAL_END  # Use train+val for training
    test_mask = dates > VAL_END
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_liquid.loc[train_mask, features].values)
    X_test = scaler.transform(df_liquid.loc[test_mask, features].values)
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    y_train = np.array(df_liquid.loc[train_mask, target].values).flatten()
    y_test = np.array(df_liquid.loc[test_mask, target].values).flatten()
    
    print(f"\nData prepared:")
    print(f"  Features:   {len(features)}")
    print(f"  Train size: {len(y_train):,}")
    print(f"  Test size:  {len(y_test):,}")
    
    # Define models
    models = {
        'Historical Mean': None,  # Special case
        'OLS': LinearRegression(),
        'Ridge (α=1.0)': Ridge(alpha=1.0),
        'Ridge (α=10.0)': Ridge(alpha=10.0),
        'Lasso (α=0.001)': Lasso(alpha=0.001),
        'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, max_depth=5, 
            n_jobs=-1, random_state=RANDOM_SEED
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100, max_depth=3,
            random_state=RANDOM_SEED
        ),
    }
    
    if HAS_XGB:
        models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=100, max_depth=3, 
            random_state=RANDOM_SEED, n_jobs=-1,
            verbosity=0
        )
    
    if HAS_LGB:
        models['LightGBM'] = lgb.LGBMRegressor(
            n_estimators=100, max_depth=3,
            random_state=RANDOM_SEED, n_jobs=-1,
            verbosity=-1
        )
    
    # Evaluate models
    results = []
    
    print("\n" + "-" * 60)
    print("Training and evaluating models...")
    print("-" * 60)
    
    for name, model in models.items():
        print(f"\n  {name}...", end=" ", flush=True)
        
        if name == 'Historical Mean':
            preds = np.full_like(y_test, y_train.mean())
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        
        r2 = r2_score(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        
        results.append({
            'Model': name,
            'R2': r2,
            'R2_pct': r2 * 100,
            'MSE': mse,
            'RMSE': np.sqrt(mse)
        })
        
        print(f"R² = {r2:.4%}")
    
    # Add Neural Network
    print(f"\n  Neural Network...", end=" ", flush=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    nn_preds = train_neural_net(X_train, y_train, X_test, len(features), device)
    nn_r2 = r2_score(y_test, nn_preds)
    nn_mse = mean_squared_error(y_test, nn_preds)
    
    results.append({
        'Model': 'Neural Network',
        'R2': nn_r2,
        'R2_pct': nn_r2 * 100,
        'MSE': nn_mse,
        'RMSE': np.sqrt(nn_mse)
    })
    print(f"R² = {nn_r2:.4%}")
    
    # Summary
    results_df = pd.DataFrame(results).sort_values('R2', ascending=False)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print()
    print(f"{'Model':<25} {'R² (%)':<12} {'RMSE':<12}")
    print("-" * 50)
    for _, row in results_df.iterrows():
        print(f"{row['Model']:<25} {row['R2_pct']:>8.4f}%    {row['RMSE']:.6f}")
    
    # Neural Network vs Best Linear
    nn_r2_val = results_df[results_df['Model'] == 'Neural Network']['R2'].values[0]
    linear_models = ['OLS', 'Ridge (α=1.0)', 'Ridge (α=10.0)', 'Lasso (α=0.001)', 'ElasticNet']
    best_linear_r2 = results_df[results_df['Model'].isin(linear_models)]['R2'].max()
    
    print()
    print(f"Neural Network vs Best Linear Model:")
    print(f"  Neural Network R²:  {nn_r2_val:.4%}")
    print(f"  Best Linear R²:     {best_linear_r2:.4%}")
    print(f"  Improvement:        {(nn_r2_val - best_linear_r2)*100:.4f} pp")
    
    # Save
    results_df.to_csv(str(RESULTS_FOLDER / "benchmark_comparison.csv"), index=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Color by model type
    colors = []
    for m in results_df['Model']:
        if 'Neural' in m:
            colors.append('#2E86AB')  # Blue for NN
        elif m == 'Historical Mean':
            colors.append('#999999')  # Gray for baseline
        elif any(x in m for x in ['XGBoost', 'LightGBM', 'Random Forest', 'Gradient']):
            colors.append('#F18F01')  # Orange for tree-based
        else:
            colors.append('#A23B72')  # Purple for linear
    
    bars = ax.barh(range(len(results_df)), results_df['R2'] * 100, color=colors)
    ax.set_yticks(range(len(results_df)))
    ax.set_yticklabels(results_df['Model'])
    ax.set_xlabel('R² (%)', fontweight='bold')
    ax.set_title('Model Comparison: Out-of-Sample R² (2017-2024)', fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E86AB', label='Neural Network'),
        Patch(facecolor='#A23B72', label='Linear Models'),
        Patch(facecolor='#F18F01', label='Tree-Based'),
        Patch(facecolor='#999999', label='Baseline')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(str(RESULTS_FOLDER / "benchmark_comparison.png"), dpi=150)
    plt.close()
    
    print(f"\n[SUCCESS] Results saved to {RESULTS_FOLDER}")
    print("=" * 60)


if __name__ == "__main__":
    main()

