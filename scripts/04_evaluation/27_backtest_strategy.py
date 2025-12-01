import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from paths import get_processed_data_path, get_results_path

# --- SETTINGS ---
# FIX: Point to the file that actually exists
FILE_PATH = get_processed_data_path("thesis_dataset_macro.parquet")
RESULTS_FOLDER = get_results_path("")
TRAIN_CUTOFF = 2016
BATCH_SIZE = 4096
EPOCHS = 30  
ENSEMBLE_MODELS = 3

RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

# --- MODEL DEF ---
class AssetPricingNet(nn.Module):
    def __init__(self, input_dim):
        super(AssetPricingNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.SiLU(), nn.Dropout(0.4),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(32, 16), nn.BatchNorm1d(16), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(16, 1)
        )
    def forward(self, x): return self.model(x)

def backtest_ml():
    print("Loading Data...")
    if not FILE_PATH.exists():
        print(f"[ERROR] File not found: {FILE_PATH}")
        return

    df = pd.read_parquet(str(FILE_PATH))
    
    # Filter Liquid Stocks
    df['size_rank'] = df.groupby('date')['mkt_cap'].rank(pct=True)
    df_liquid = df[df['size_rank'] > 0.3].copy()
    
    # Prepare Features
    exclude = ['permno', 'date', 'ticker', 'ret', 'future_ret', 'prc', 'shrout', 'vol', 'bid', 'ask', 'siccd', 'size_rank', 'year', 'match_year', 'fyear']
    numeric_cols = df_liquid.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in exclude]
    target = 'future_ret'
    
    print(f"Backtesting with {len(features)} features...")

    # Sanitization
    print("Sanitizing...")
    for col in features:
        df_liquid[col] = pd.to_numeric(df_liquid[col], errors='coerce')
        df_liquid[col] = df_liquid[col].replace([np.inf, -np.inf], np.nan)
        median_val = df_liquid[col].median()
        df_liquid[col] = df_liquid[col].fillna(0 if pd.isna(median_val) else median_val)
        lower, upper = df_liquid[col].quantile(0.01), df_liquid[col].quantile(0.99)
        df_liquid[col] = df_liquid[col].clip(lower, upper)
    
    df_liquid = df_liquid.dropna(subset=[target])
    
    # Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(df_liquid[features].values)
    y = df_liquid[target].values.reshape(-1, 1)
    X = np.nan_to_num(X, nan=0.0)
    
    # Split
    dates = df_liquid['date'].dt.year.values
    train_mask = dates <= TRAIN_CUTOFF
    test_mask = dates > TRAIN_CUTOFF
    
    # Tracking for Backtest
    test_dates = df_liquid.loc[test_mask, 'date'].values
    test_returns = df_liquid.loc[test_mask, 'future_ret'].values
    
    X_train_t = torch.tensor(X[train_mask], dtype=torch.float32)
    y_train_t = torch.tensor(y[train_mask], dtype=torch.float32)
    X_test_t = torch.tensor(X[test_mask], dtype=torch.float32)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training Strategy Model on {device}...")
    
    # --- TRAIN & PREDICT ---
    ensemble_preds = np.zeros((len(X_test_t), ENSEMBLE_MODELS))
    
    for i in range(ENSEMBLE_MODELS):
        print(f"   Training Ensemble Member {i+1}...")
        model = AssetPricingNet(len(features)).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
        criterion = nn.MSELoss()
        
        loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
        
        model.train()
        for epoch in range(EPOCHS):
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
        
        model.eval()
        with torch.no_grad():
            ensemble_preds[:, i] = model(X_test_t.to(device)).cpu().numpy().flatten()

    # Average Predictions
    final_preds = np.mean(ensemble_preds, axis=1)
    
    # --- BACKTEST LOGIC ---
    print("Constructing Portfolio...")
    
    backtest_df = pd.DataFrame({
        'date': test_dates,
        'return': test_returns,
        'prediction': final_preds
    })
    
    # Quintile Sort
    backtest_df['rank'] = backtest_df.groupby('date')['prediction'].rank(pct=True)
    
    long_port = backtest_df[backtest_df['rank'] >= 0.8].groupby('date')['return'].mean()
    short_port = backtest_df[backtest_df['rank'] <= 0.2].groupby('date')['return'].mean()
    
    strategy_ret = long_port - short_port
    
    # Metrics
    annual_ret = strategy_ret.mean() * 12
    annual_vol = strategy_ret.std() * np.sqrt(12)
    sharpe = annual_ret / annual_vol
    
    print(f"\n--- ML STRATEGY RESULTS (2017-2024) ---")
    print(f"Annualized Return:   {annual_ret:.2%}")
    print(f"Annualized Volatility: {annual_vol:.2%}")
    print(f"Sharpe Ratio:        {sharpe:.2f}")
    
    # Plot
    cum_ret = (1 + strategy_ret).cumprod()
    plt.figure(figsize=(10, 6))
    cum_ret.plot(label=f'Neural Net Strategy (SR={sharpe:.2f})', color='green', linewidth=2)
    plt.title("Cumulative Returns: Neural Net Strategy vs Market")
    plt.ylabel("Growth of $1 Investment")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(str(RESULTS_FOLDER / "ml_strategy_performance.png"))
    print(f"[SUCCESS] Chart saved.")

if __name__ == "__main__":
    backtest_ml()