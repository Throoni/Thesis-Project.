"""
Build Momentum Features from CRSP Monthly Data

Calculates standard momentum signals following Jegadeesh & Titman (1993):
- mom1m: Short-term reversal (t-1)
- mom6m: Intermediate momentum (t-2 to t-6, skip t-1)
- mom12m: Standard momentum (t-2 to t-12, skip t-1)
- mom36m: Long-term reversal (t-13 to t-36)
- chmom: Change in momentum (acceleration)

Author: Thesis Project
Date: December 2024
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from paths import get_raw_data_path, get_processed_data_path

# --- SETTINGS ---
INPUT_FILE = get_raw_data_path("crsp_stocks_full.parquet")
OUTPUT_FILE = get_processed_data_path("features_momentum.parquet")

def build_momentum():
    print("=" * 60)
    print("BUILDING MOMENTUM FEATURES")
    print("=" * 60)
    
    print("\nLoading Monthly CRSP Data...")
    df = pd.read_parquet(str(INPUT_FILE))
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['permno', 'date'])
    
    print("Calculating Momentum Signals...")
    
    # Log Returns for additive calculations
    df['log_ret'] = np.log(1 + df['ret'].fillna(0))
    grouped = df.groupby('permno')['log_ret']
    
    # 1. Short-Term Reversal (t-1)
    print("  - mom1m: Short-term reversal")
    df['mom1m'] = grouped.shift(1)
    
    # 2. Standard Momentum (t-12 to t-2)
    # Cumulative return of months t-12 through t-2 (skipping t-1 for microstructure)
    print("  - mom12m: Standard momentum (skip t-1)")
    df['mom12m'] = grouped.shift(2).rolling(11).sum()
    
    # 3. Intermediate Momentum (t-6 to t-2)
    print("  - mom6m: Intermediate momentum")
    df['mom6m'] = grouped.shift(2).rolling(5).sum()
    
    # 4. Long-Term Reversal (t-36 to t-13)
    print("  - mom36m: Long-term reversal")
    df['mom36m'] = grouped.shift(13).rolling(24).sum()
    
    # 5. Change in Momentum (Acceleration)
    print("  - chmom: Momentum acceleration")
    recent_mom = grouped.shift(1).rolling(6).sum()
    old_mom = grouped.shift(7).rolling(6).sum()
    df['chmom'] = recent_mom - old_mom
    
    # Select columns to save
    keep_cols = ['permno', 'date', 'mom1m', 'mom6m', 'mom12m', 'mom36m', 'chmom']
    df_final = df[keep_cols].dropna(subset=['mom12m'])
    
    # Ensure output directory exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(str(OUTPUT_FILE))
    
    print()
    print("=" * 60)
    print(f"[SUCCESS] Momentum features saved")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Rows: {len(df_final):,}")
    print("=" * 60)

if __name__ == "__main__":
    build_momentum()