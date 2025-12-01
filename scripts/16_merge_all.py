import pandas as pd
import numpy as np
import os

# --- SETTINGS ---
DATA_FOLDER = "raw_data"
PROCESSED_FOLDER = "processed_data"
OUTPUT_FILE = "processed_data/thesis_dataset_final.parquet"

def merge_all_features():
    print("Loading Base CRSP Data (The Skeleton)...")
    # We start with the raw CRSP file because it has the 'ret' (Target Variable)
    df = pd.read_parquet(os.path.join(DATA_FOLDER, "crsp_stocks_full.parquet"))
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['permno', 'date'])
    
    # 1. Create Target: Next Month's Return (t+1)
    print("Constructing Target Variable (t+1)...")
    df['future_ret'] = df.groupby('permno')['ret'].shift(-1)
    
    # 2. Merge Momentum (Module 1)
    print("Merging Momentum...")
    path = os.path.join(PROCESSED_FOLDER, "features_momentum.parquet")
    if os.path.exists(path):
        df_mom = pd.read_parquet(path)
        df = df.merge(df_mom, on=['permno', 'date'], how='left')
        
    # 3. Merge Daily Risk (Module 2)
    print("Merging Daily Risk (Beta, MaxRet)...")
    path = os.path.join(PROCESSED_FOLDER, "features_risk_daily.parquet")
    if os.path.exists(path):
        df_risk = pd.read_parquet(path)
        df = df.merge(df_risk, on=['permno', 'date'], how='left')

    # 4. Merge Liquidity (Module 3)
    print("Merging Liquidity...")
    path = os.path.join(PROCESSED_FOLDER, "features_liquidity.parquet")
    if os.path.exists(path):
        df_liq = pd.read_parquet(path)
        df = df.merge(df_liq, on=['permno', 'date'], how='left')

    # 5. Merge Fundamentals (Module 4)
    print("Merging Fundamentals...")
    path = os.path.join(PROCESSED_FOLDER, "features_fundamental.parquet")
    if os.path.exists(path):
        df_fund = pd.read_parquet(path)
        df = df.merge(df_fund, on=['permno', 'date'], how='left')

    # --- CLEANING & IMPUTATION ---
    print("Cleaning Data...")
    
    # Filter 1: Price > $1 (Avoid microstructure noise)
    df['prc'] = df['prc'].abs()
    df = df[df['prc'] >= 1]
    
    # Filter 2: Must have a valid Target (Future Return)
    df = df.dropna(subset=['future_ret'])
    
    # Filter 3: Median Imputation for Missing Features
    # (Standard academic practice: if missing, assume average)
    predictors = [
        'mom1m', 'mom12m', 'mom36m', 'chmom',       # Momentum
        'retvol', 'maxret', 'zero_trades',          # Risk
        'turnover', 'dolvol', 'spread',             # Liquidity
        'bm', 'ep', 'sp'                            # Fundamentals
    ]
    
    # Only impute columns that actually exist (in case a script failed)
    valid_predictors = [c for c in predictors if c in df.columns]
    
    print(f"Imputing {len(valid_predictors)} predictors (Monthly Median)...")
    for col in valid_predictors:
        # Group by DATE to calculate that month's median
        df[col] = df.groupby('date')[col].transform(lambda x: x.fillna(x.median()))
    
    # Final Drop: If a column is STILL empty (e.g., entire month missing), drop row
    df_clean = df.dropna(subset=valid_predictors)
    
    # Save
    df_clean.to_parquet(OUTPUT_FILE)
    
    print("-" * 30)
    print(f"FINAL DATASET SAVED: {OUTPUT_FILE}")
    print(f"Total Observations: {len(df_clean):,}")
    print(f"Unique Stocks: {df_clean['permno'].nunique():,}")
    print(f"Predictors: {valid_predictors}")
    print("-" * 30)

if __name__ == "__main__":
    merge_all_features()