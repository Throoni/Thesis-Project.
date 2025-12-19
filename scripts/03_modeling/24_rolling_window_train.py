"""
Rolling Window Training and Evaluation

Trains models on expanding windows to:
1. Mimic real-world trading conditions
2. Test stability across different market regimes
3. Produce more reliable out-of-sample estimates

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

sys.path.append(str(Path(__file__).parent.parent / "utils"))
from paths import get_processed_data_path, get_results_path

# Settings
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

FILE_PATH = get_processed_data_path("thesis_dataset_macro.parquet")
RESULTS_FOLDER = get_results_path("")

# Rolling window settings
INITIAL_TRAIN_YEARS = 10  # Start with 10 years of training data
WINDOW_TYPE = 'expanding'  # 'expanding' uses all prior data


class AssetPricingNet(nn.Module):
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


def train_model(X_train, y_train, n_features, device, epochs=30):
    """Train a single model and return it."""
    model = AssetPricingNet(n_features).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)
    criterion = nn.MSELoss()
    
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=4096, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
    
    return model


def main():
    print("=" * 60)
    print("ROLLING WINDOW TRAINING")
    print("=" * 60)
    print(f"\nWindow Type: {WINDOW_TYPE}")
    print(f"Initial Training Years: {INITIAL_TRAIN_YEARS}")
    
    # Load data
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
    for col in features:
        df_liquid[col] = pd.to_numeric(df_liquid[col], errors='coerce')
        df_liquid[col] = df_liquid[col].replace([np.inf, -np.inf], np.nan)
        median_val = df_liquid[col].median()
        df_liquid[col] = df_liquid[col].fillna(0 if pd.isna(median_val) else median_val)
        lower, upper = df_liquid[col].quantile(0.01), df_liquid[col].quantile(0.99)
        df_liquid[col] = df_liquid[col].clip(lower, upper)
    
    df_liquid = df_liquid.dropna(subset=[target])
    df_liquid['year'] = df_liquid['date'].dt.year
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Get unique years
    years = sorted(df_liquid['year'].unique())
    start_year = years[0]
    first_test_year = start_year + INITIAL_TRAIN_YEARS
    
    print(f"\nData years: {years[0]} - {years[-1]}")
    print(f"First test year: {first_test_year}")
    
    # Rolling window evaluation
    results = []
    test_years = [y for y in years if y >= first_test_year]
    
    print(f"\nEvaluating {len(test_years)} years...")
    print("-" * 60)
    
    for i, test_year in enumerate(test_years):
        # Determine training window (expanding)
        train_start = start_year
        train_end = test_year - 1
        
        print(f"\nPeriod {i+1}/{len(test_years)}: Train {train_start}-{train_end}, Test {test_year}")
        
        # Split data
        train_mask = (df_liquid['year'] >= train_start) & (df_liquid['year'] <= train_end)
        test_mask = df_liquid['year'] == test_year
        
        if train_mask.sum() < 1000 or test_mask.sum() < 100:
            print(f"  Skipping: insufficient data")
            continue
        
        # Scale (fit on training only)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(df_liquid.loc[train_mask, features].values)
        X_test = scaler.transform(df_liquid.loc[test_mask, features].values)
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)
        
        y_train = df_liquid.loc[train_mask, target].values.reshape(-1, 1)
        y_test = np.array(df_liquid.loc[test_mask, target].values).flatten()
        
        # Train model
        torch.manual_seed(RANDOM_SEED + i)  # Different seed per period
        model = train_model(X_train, y_train, len(features), device)
        
        # Predict
        model.eval()
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = model(X_test_t).cpu().numpy().flatten()
        
        # Evaluate
        r2 = r2_score(y_test, preds)
        
        print(f"  Train: {train_mask.sum():,} | Test: {test_mask.sum():,} | R²: {r2:.4%}")
        
        results.append({
            'test_year': test_year,
            'train_start': train_start,
            'train_end': train_end,
            'train_size': train_mask.sum(),
            'test_size': test_mask.sum(),
            'r2': r2
        })
    
    # Summary
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 60)
    print("ROLLING WINDOW RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nAverage R² across periods: {results_df['r2'].mean():.4%}")
    print(f"Std Dev of R²:             {results_df['r2'].std():.4%}")
    print(f"Min R²:                    {results_df['r2'].min():.4%} ({results_df.loc[results_df['r2'].idxmin(), 'test_year']})")
    print(f"Max R²:                    {results_df['r2'].max():.4%} ({results_df.loc[results_df['r2'].idxmax(), 'test_year']})")
    print(f"\nPositive R² years:         {(results_df['r2'] > 0).sum()}/{len(results_df)}")
    
    # Save results
    results_df.to_csv(str(RESULTS_FOLDER / "rolling_window_results.csv"), index=False)
    
    # Plot R² over time
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart by year
    colors = ['#2E86AB' if r > 0 else '#A23B72' for r in results_df['r2']]
    axes[0].bar(results_df['test_year'], results_df['r2'] * 100, color=colors)
    axes[0].axhline(y=results_df['r2'].mean() * 100, color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {results_df["r2"].mean()*100:.2f}%')
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0].set_xlabel('Test Year', fontweight='bold')
    axes[0].set_ylabel('R² (%)', fontweight='bold')
    axes[0].set_title('Out-of-Sample R² by Year (Expanding Window)', fontweight='bold')
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Cumulative average
    cumavg = results_df['r2'].expanding().mean() * 100
    axes[1].plot(results_df['test_year'], cumavg, marker='o', color='#2E86AB', linewidth=2)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_xlabel('Test Year', fontweight='bold')
    axes[1].set_ylabel('Cumulative Average R² (%)', fontweight='bold')
    axes[1].set_title('Cumulative Average R² Over Time', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(RESULTS_FOLDER / "rolling_window_r2.png"), dpi=150)
    plt.close()
    
    print(f"\n[SUCCESS] Results saved to {RESULTS_FOLDER}")
    print("=" * 60)


if __name__ == "__main__":
    main()

