import pandas as pd
import numpy as np
import os

# --- SETTINGS ---
DATA_FOLDER = "raw_data"
OUTPUT_FOLDER = "processed_data"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def load_and_process():
    print("Loading CRSP data...")
    # Load the CRSP file
    df = pd.read_parquet(os.path.join(DATA_FOLDER, "crsp_stocks_full.parquet"))
    
    # CRITICAL: Sort by Stock and Date so rolling windows work correctly
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['permno', 'date'])
    
    print("Calculating Price Features...")
    
    # 1. Market Cap (Size)
    df['prc'] = df['prc'].abs() 
    df['mkt_cap'] = np.log(df['prc'] * df['shrout'])
    
    # 2. Momentum & Volatility
    # We calculate Log Returns for additive math
    df['log_ret'] = np.log(1 + df['ret'].fillna(0))
    
    # Create the Group Object
    grouped = df.groupby('permno')['log_ret']
    
    # --- FIX: Use reset_index(level=0, drop=True) to align indices ---
    
    # Momentum (t-12 to t-1): Sum of last 11 months, shifted by 1
    print("   - Calculating Momentum...")
    df['mom12m'] = grouped.shift(1).rolling(11).sum().reset_index(level=0, drop=True)
    
    # Short-Term Reversal (t-1)
    df['mom1m'] = grouped.shift(1).reset_index(level=0, drop=True)
    
    # Volatility (12-month rolling std dev)
    print("   - Calculating Volatility...")
    df['volatility_12m'] = grouped.rolling(12).std().reset_index(level=0, drop=True)
    
    # Liquidity (Turnover)
    # Handle division by zero if shares are missing
    turnover_ratio = (df['vol'] / df['shrout']).fillna(0)
    df['turnover'] = np.log(1 + turnover_ratio)
    
    # --- MANUAL MERGE (Compustat) ---
    comp_path = os.path.join(DATA_FOLDER, "compustat_annual.parquet")
    
    if os.path.exists(comp_path):
        print("Merging Compustat Data (Manual Link via Ticker)...")
        df_fund = pd.read_parquet(comp_path)
        df_fund['datadate'] = pd.to_datetime(df_fund['datadate'])
        
        # Calculate Book Value (Shevlin 1990 method)
        df_fund['be'] = df_fund['seq'] + df_fund['txditc'].fillna(0) - df_fund['pstk'].fillna(0)
        df_fund['be'] = df_fund['be'].where(df_fund['be'] > 0)
        df_fund['earnings'] = df_fund['ib']
        
        # Clean Tickers for merging
        df['ticker'] = df['ticker'].astype(str).str.upper()
        df_fund['tic'] = df_fund['tic'].astype(str).str.upper()
        
        # Match Fiscal Year to Following Calendar Year
        df['match_year'] = df['date'].dt.year
        df_fund['match_year'] = df_fund['datadate'].dt.year + 1 
        
        # Deduplicate (Take last report per ticker/year)
        df_fund_unique = df_fund.sort_values('datadate').groupby(['tic', 'match_year']).last().reset_index()
        
        # Merge
        df_merged = pd.merge(
            df, 
            df_fund_unique[['tic', 'match_year', 'be', 'earnings']], 
            left_on=['ticker', 'match_year'], 
            right_on=['tic', 'match_year'], 
            how='left'
        )
        
        # Calculate Valuation Ratios
        market_equity = df_merged['prc'] * df_merged['shrout']
        # Scale Compustat (Millions) to match CRSP (Thousands)
        df_merged['bm'] = (df_merged['be'] * 1000) / market_equity 
        df_merged['ep'] = (df_merged['earnings'] * 1000) / market_equity
        
        df = df_merged
        print(f"   - Merged! Rows with valid Book-to-Market: {df['bm'].notna().sum():,}")
        
    else:
        print("Skipping Compustat merge (File not found)")

    # Final Clean
    # We require Momentum, Return, and Market Cap to exist.
    # We do NOT drop if BM/EP are missing (we can handle that in the model)
    print("Finalizing dataset...")
    df_clean = df.dropna(subset=['ret', 'mom12m', 'mkt_cap'])
    
    # Select only the columns we need for the Thesis
    keep_cols = [
        'permno', 'date', 'ticker', 'ret', 
        'mkt_cap', 'mom1m', 'mom12m', 'volatility_12m', 'turnover',
        'bm', 'ep'
    ]
    # Only keep columns that actually exist (in case Compustat failed)
    final_cols = [c for c in keep_cols if c in df_clean.columns]
    df_clean = df_clean[final_cols]

    save_path = os.path.join(OUTPUT_FOLDER, "stock_predictors.parquet")
    df_clean.to_parquet(save_path)
    print(f"Processed data saved to {save_path}.")
    print(f"Total Observations: {len(df_clean):,}")

if __name__ == "__main__":
    load_and_process()