import pandas as pd
import numpy as np
import os

INPUT_FILE = "raw_data/crsp_stocks_full.parquet"
OUTPUT_FOLDER = "processed_data"

def build_liquidity():
    print("Loading CRSP Monthly Data...")
    df = pd.read_parquet(INPUT_FILE)
    df['date'] = pd.to_datetime(df['date'])
    
    print("Calculating Liquidity Signals...")
    
    # Handle negative prices (bid/ask average)
    df['prc'] = df['prc'].abs()
    
    # 1. Share Turnover (Log)
    # Measures trading activity relative to size
    # We use log(1 + x) to handle skewness
    df['turnover'] = np.log(1 + (df['vol'] / df['shrout']))
    
    # 2. Dollar Volume (Log)
    # Proxy for institutional attention / limits to arbitrage
    df['dolvol'] = np.log(1 + (df['prc'] * df['vol']))
    
    # 3. Bid-Ask Spread (Illiquidity)
    # (Ask - Bid) / Midpoint
    midpoint = (df['ask'] + df['bid']) / 2
    df['spread'] = (df['ask'] - df['bid']) / midpoint
    
    # 4. Amihud Illiquidity (Monthly Proxy)
    # Abs(Ret) / Dollar Volume
    # Note: The "true" Amihud requires daily data averaging. 
    # This is a monthly approximation often used when daily is too heavy.
    # Since we have daily data running in script 13, we could do it there, 
    # but this is a good backup if script 13 fails.
    df['amihud_monthly'] = df['ret'].abs() / (df['prc'] * df['vol'])
    
    # Select and Save
    keep_cols = ['permno', 'date', 'turnover', 'dolvol', 'spread', 'amihud_monthly']
    
    # We only keep rows that have at least one liquidity metric
    df_final = df[keep_cols].dropna(thresh=3) 
    
    save_path = os.path.join(OUTPUT_FOLDER, "features_liquidity.parquet")
    df_final.to_parquet(save_path)
    print(f"[SUCCESS] Liquidity features saved: {len(df_final):,} rows.")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    build_liquidity()