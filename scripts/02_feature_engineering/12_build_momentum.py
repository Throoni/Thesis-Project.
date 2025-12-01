import pandas as pd
import numpy as np
import os

# --- SETTINGS ---
INPUT_FILE = "raw_data/crsp_stocks_full.parquet"
OUTPUT_FOLDER = "processed_data"

def build_momentum():
    print("Loading Monthly CRSP Data...")
    df = pd.read_parquet(INPUT_FILE)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['permno', 'date'])
    
    print("Calculating Momentum Signals...")
    
    # Log Returns for additive calculations
    df['log_ret'] = np.log(1 + df['ret'].fillna(0))
    grouped = df.groupby('permno')['log_ret']
    
    # 1. Short-Term Reversal (t-1)
    df['mom1m'] = grouped.shift(1)
    
    # 2. Standard Momentum (t-12 to t-2)
    # Cumulative return of months t-12 through t-2 (skipping t-1)
    df['mom12m'] = grouped.shift(2).rolling(11).sum()
    
    # 3. Intermediate Momentum (t-6 to t-2)
    df['mom6m'] = grouped.shift(2).rolling(5).sum()
    
    # 4. Long-Term Reversal (t-36 to t-13)
    df['mom36m'] = grouped.shift(13).rolling(24).sum()
    
    # 5. Change in Momentum (Acceleration)
    # Difference between recent momentum (months 1-6) and older momentum (months 7-12)
    # Logic: (Avg ret t-6...t-1) - (Avg ret t-12...t-7)
    # We use shift(1) to ensure we don't use current month
    recent_mom = grouped.shift(1).rolling(6).sum()
    old_mom = grouped.shift(7).rolling(6).sum()
    df['chmom'] = recent_mom - old_mom
    
    # Select columns to save
    keep_cols = ['permno', 'date', 'mom1m', 'mom6m', 'mom12m', 'mom36m', 'chmom']
    df_final = df[keep_cols].dropna(subset=['mom12m'])
    
    save_path = os.path.join(OUTPUT_FOLDER, "features_momentum.parquet")
    df_final.to_parquet(save_path)
    print(f"[SUCCESS] Momentum features saved: {len(df_final):,} rows.")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    build_momentum()