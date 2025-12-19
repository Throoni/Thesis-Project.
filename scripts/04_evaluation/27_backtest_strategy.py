"""
Backtest Strategy for Neural Network Stock Predictions

This script evaluates the economic significance of the model by:
1. Constructing quintile portfolios based on predictions
2. Computing long-short strategy returns
3. Accounting for transaction costs
4. Comparing to a market benchmark

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
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from paths import get_processed_data_path, get_results_path

# --- REPRODUCIBILITY ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# --- SETTINGS ---
FILE_PATH = get_processed_data_path("thesis_dataset_macro.parquet")
RESULTS_FOLDER = get_results_path("")

# Time Splits (must match training script)
TRAIN_END = 2013      # Training: 1996-2013
VAL_END = 2016        # Validation: 2014-2016

BATCH_SIZE = 4096
EPOCHS = 30  
ENSEMBLE_MODELS = 3

# Transaction cost assumptions
TRANSACTION_COST_BPS = 10  # 10 basis points per trade (conservative)

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
    print("=" * 60)
    print("NEURAL NETWORK STRATEGY BACKTEST")
    print("=" * 60)
    
    print("\nLoading Data...")
    if not FILE_PATH.exists():
        print(f"[ERROR] File not found: {FILE_PATH}")
        return

    df = pd.read_parquet(str(FILE_PATH))
    
    # Filter Liquid Stocks (top 70% by market cap)
    df['size_rank'] = df.groupby('date')['mkt_cap'].rank(pct=True)
    df_liquid = df[df['size_rank'] > 0.3].copy()
    
    # Prepare Features
    exclude = ['permno', 'date', 'ticker', 'ret', 'future_ret', 'prc', 'shrout', 
               'vol', 'bid', 'ask', 'siccd', 'size_rank', 'year', 'match_year', 'fyear']
    numeric_cols = df_liquid.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in exclude]
    target = 'future_ret'
    
    print(f"Features: {len(features)}")

    # Sanitization
    print("Sanitizing data...")
    for col in features:
        df_liquid[col] = pd.to_numeric(df_liquid[col], errors='coerce')
        df_liquid[col] = df_liquid[col].replace([np.inf, -np.inf], np.nan)
        median_val = df_liquid[col].median()
        df_liquid[col] = df_liquid[col].fillna(0 if pd.isna(median_val) else median_val)
        lower, upper = df_liquid[col].quantile(0.01), df_liquid[col].quantile(0.99)
        df_liquid[col] = df_liquid[col].clip(lower, upper)
    
    df_liquid = df_liquid.dropna(subset=[target])
    
    # PROPER TRAIN/VAL/TEST SPLIT (matching training script)
    dates = df_liquid['date'].dt.year.values
    train_mask = dates <= TRAIN_END
    val_mask = (dates > TRAIN_END) & (dates <= VAL_END)
    test_mask = dates > VAL_END
    
    # Use train + validation for training (val was used for early stopping, not cheating)
    train_val_mask = dates <= VAL_END
    
    print(f"\nData Split:")
    print(f"  Training: {train_val_mask.sum():,} obs (1996-{VAL_END})")
    print(f"  Testing:  {test_mask.sum():,} obs ({VAL_END+1}-2024)")
    
    # Scaling - Fit ONLY on training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_liquid.loc[train_val_mask, features].values)
    X_test = scaler.transform(df_liquid.loc[test_mask, features].values)
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    y_train = df_liquid.loc[train_val_mask, target].values.reshape(-1, 1)
    
    # Tracking for Backtest
    test_dates = df_liquid.loc[test_mask, 'date'].values
    test_returns = df_liquid.loc[test_mask, 'future_ret'].values
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nTraining on {device}...")
    
    # --- TRAIN ENSEMBLE & PREDICT ---
    ensemble_preds = np.zeros((len(X_test_t), ENSEMBLE_MODELS))
    
    for i in range(ENSEMBLE_MODELS):
        torch.manual_seed(RANDOM_SEED + i)
        print(f"   Training Ensemble Member {i+1}/{ENSEMBLE_MODELS}...")
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
    print("\nConstructing Portfolios...")
    
    backtest_df = pd.DataFrame({
        'date': test_dates,
        'return': test_returns,
        'prediction': final_preds
    })
    
    # Quintile Sort each month
    backtest_df['rank'] = backtest_df.groupby('date')['prediction'].rank(pct=True)
    
    # Long top quintile, short bottom quintile
    long_port = backtest_df[backtest_df['rank'] >= 0.8].groupby('date')['return'].mean()
    short_port = backtest_df[backtest_df['rank'] <= 0.2].groupby('date')['return'].mean()
    
    # Market benchmark (equal-weighted portfolio)
    market_ret = backtest_df.groupby('date')['return'].mean()
    
    strategy_ret_gross = long_port - short_port
    
    # --- TRANSACTION COSTS ---
    # Estimate turnover: assume ~40% monthly turnover for each leg
    # Cost = turnover * 2 (buy + sell) * cost per trade
    monthly_turnover = 0.4  # Conservative estimate
    monthly_cost = monthly_turnover * 2 * (TRANSACTION_COST_BPS / 10000)
    
    strategy_ret_net = strategy_ret_gross - monthly_cost
    
    # --- METRICS (GROSS) ---
    annual_ret_gross = strategy_ret_gross.mean() * 12
    annual_vol_gross = strategy_ret_gross.std() * np.sqrt(12)
    sharpe_gross = annual_ret_gross / annual_vol_gross if annual_vol_gross > 0 else 0
    
    # --- METRICS (NET) ---
    annual_ret_net = strategy_ret_net.mean() * 12
    annual_vol_net = strategy_ret_net.std() * np.sqrt(12)
    sharpe_net = annual_ret_net / annual_vol_net if annual_vol_net > 0 else 0
    
    # --- MARKET BENCHMARK ---
    market_annual_ret = market_ret.mean() * 12
    market_annual_vol = market_ret.std() * np.sqrt(12)
    market_sharpe = market_annual_ret / market_annual_vol if market_annual_vol > 0 else 0
    
    print(f"\n" + "=" * 60)
    print(f"STRATEGY RESULTS ({VAL_END+1}-2024)")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'Gross':<15} {'Net of Costs':<15}")
    print("-" * 55)
    print(f"{'Annualized Return':<25} {annual_ret_gross:>12.2%}   {annual_ret_net:>12.2%}")
    print(f"{'Annualized Volatility':<25} {annual_vol_gross:>12.2%}   {annual_vol_net:>12.2%}")
    print(f"{'Sharpe Ratio':<25} {sharpe_gross:>12.2f}   {sharpe_net:>12.2f}")
    print()
    print(f"Transaction Cost Assumption: {TRANSACTION_COST_BPS} bps per trade")
    print(f"Estimated Monthly Turnover:  {monthly_turnover*100:.0f}%")
    print()
    print(f"Market Benchmark (Equal-Weighted):")
    print(f"  Annualized Return:   {market_annual_ret:.2%}")
    print(f"  Sharpe Ratio:        {market_sharpe:.2f}")
    
    # Save returns for further analysis
    returns_df = pd.DataFrame({
        'date': strategy_ret_gross.index,
        'strategy_return_gross': strategy_ret_gross.values,
        'strategy_return_net': strategy_ret_net.values,
        'market_return': market_ret.reindex(strategy_ret_gross.index).values
    })
    returns_df.to_csv(str(RESULTS_FOLDER / "backtest_returns.csv"), index=False)
    
    # Plot
    cum_ret_gross = (1 + strategy_ret_gross).cumprod()
    cum_ret_net = (1 + strategy_ret_net).cumprod()
    cum_market = (1 + market_ret).cumprod()
    
    plt.figure(figsize=(12, 6))
    cum_ret_gross.plot(label=f'NN Strategy Gross (SR={sharpe_gross:.2f})', color='#2E86AB', linewidth=2)
    cum_ret_net.plot(label=f'NN Strategy Net (SR={sharpe_net:.2f})', color='#2E86AB', linewidth=2, linestyle='--')
    cum_market.plot(label=f'Market EW (SR={market_sharpe:.2f})', color='#A23B72', linewidth=2)
    
    plt.title(f"Cumulative Returns: Neural Net Strategy vs Market ({VAL_END+1}-2024)")
    plt.ylabel("Growth of $1 Investment")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(str(RESULTS_FOLDER / "ml_strategy_performance.png"), dpi=150)
    plt.close()
    
    print(f"\n[SUCCESS] Results saved to {RESULTS_FOLDER}")
    print("=" * 60)

if __name__ == "__main__":
    backtest_ml()