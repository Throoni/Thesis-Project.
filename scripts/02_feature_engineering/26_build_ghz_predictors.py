import pandas as pd
import numpy as np
import os

# --- SETTINGS ---
STOCK_FILE = "raw_data/crsp_stocks_full.parquet"
COMP_FILE = "raw_data/compustat_annual.parquet"
OUTPUT_FILE = "processed_data/ghz_predictors_87.parquet"

def build_ghz():
    print("Loading CRSP Data...")
    df = pd.read_parquet(STOCK_FILE)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['permno', 'date'])
    
    # 1. PRICE & LIQUIDITY SIGNALS (CRSP)
    print("Calculating Price/Volume Signals...")
    df['prc'] = df['prc'].abs()
    # Calculate Market Cap
    df['mkt_cap'] = df['prc'] * df['shrout']
    df['log_ret'] = np.log(1 + df['ret'].fillna(0))
    
    # --- MOMENTUM & VOLATILITY ---
    print("   - Calculating Momentum & Reversal...")
    df['mom1m'] = df.groupby('permno')['log_ret'].transform(lambda x: x.shift(1))
    df['mom12m'] = df.groupby('permno')['log_ret'].transform(lambda x: x.shift(1).rolling(11).sum())
    df['mom36m'] = df.groupby('permno')['log_ret'].transform(lambda x: x.shift(13).rolling(24).sum())
    df['mom6m'] = df.groupby('permno')['log_ret'].transform(lambda x: x.shift(1).rolling(5).sum())
    
    # Momentum Change
    mom6_lag = df.groupby('permno')['log_ret'].transform(lambda x: x.shift(7).rolling(6).sum())
    df['chmom'] = df['mom6m'] - mom6_lag
    
    print("   - Calculating Volatility & Liquidity...")
    df['volatility'] = df.groupby('permno')['log_ret'].transform(lambda x: x.rolling(12).std())
    
    # Liquidity
    vol_ratio = (df['vol'] / df['shrout']).replace(0, np.nan)
    df['turnover'] = np.log(1 + vol_ratio.fillna(0))
    
    dol_vol = (df['prc'] * df['vol']).replace(0, np.nan)
    df['dolvol'] = np.log(1 + dol_vol.fillna(0))
    
    midpoint = (df['ask'] + df['bid']) / 2
    df['ba_spread'] = (df['ask'] - df['bid']) / midpoint
    
    # Seasonality
    print("   - Calculating Seasonality...")
    for y in range(1, 6):
        df[f'season_{y}y'] = df.groupby('permno')['ret'].transform(lambda x: x.shift(12 * y))

    # 2. FUNDAMENTAL SIGNALS (Compustat)
    print("Processing Fundamentals...")
    if not os.path.exists(COMP_FILE):
        print(f"[ERROR] Compustat file not found at {COMP_FILE}")
        return

    fund = pd.read_parquet(COMP_FILE)
    fund['datadate'] = pd.to_datetime(fund['datadate'])
    
    # Clean zeros
    for col in fund.columns:
        if fund[col].dtype in ['float64', 'float32', 'int64']:
            fund[col] = fund[col].replace(0, np.nan)

    # A. Valuation
    if 'txditc' in fund:
        fund['bm'] = (fund['seq'] + fund['txditc'].fillna(0) - fund['pstk'].fillna(0)) 
    else:
        fund['bm'] = fund['seq']
        
    fund['ep'] = fund['ib'] 
    fund['cp'] = fund['che'] 
    fund['sp'] = fund['sale'] 
    fund['dy'] = fund['dv'] 

    # B. Solvency & Liquidity
    fund['lev'] = (fund['dltt'] + fund['dlc']) / fund['seq']
    fund['debt_assets'] = (fund['dltt'] + fund['dlc']) / fund['at']
    fund['currat'] = fund['act'] / fund['lct']
    fund['quick'] = (fund['act'] - fund['invt']) / fund['lct']
    fund['cash_ratio'] = fund['che'] / fund['lct']
    
    # C. Profitability
    if 'cogs' in fund.columns:
        fund['oper_prof'] = (fund['sale'] - fund['cogs']) / fund['at']
        fund['gross_prof'] = (fund['sale'] - fund['cogs'])
        fund['inv_turn'] = fund['cogs'] / fund['invt']
    else:
        print("[WARNING] 'cogs' missing. Filling with NaN.")
        fund['oper_prof'] = np.nan
        fund['gross_prof'] = np.nan
        fund['inv_turn'] = np.nan

    fund['roeq'] = fund['ib'] / fund['seq']
    fund['roaq'] = fund['ib'] / fund['at']
    fund['profit_margin'] = fund['ib'] / fund['sale']
    
    # D. Efficiency
    fund['at_turn'] = fund['sale'] / fund['at']
    fund['rect_turn'] = fund['sale'] / fund['rect']
    
    # E. Investment / Growth
    fund = fund.sort_values(['gvkey', 'datadate'])
    fund['agr'] = fund.groupby('gvkey')['at'].pct_change(fill_method=None)
    fund['sgr'] = fund.groupby('gvkey')['sale'].pct_change(fill_method=None)
    fund['egr'] = fund.groupby('gvkey')['seq'].pct_change(fill_method=None)
    fund['lgr'] = fund.groupby('gvkey')['dltt'].pct_change(fill_method=None)
    fund['inv_growth'] = fund.groupby('gvkey')['invt'].pct_change(fill_method=None)
    
    # F. Intangibles
    fund['rd_sale'] = fund['xrd'] / fund['sale']
    fund['rd_assets'] = fund['xrd'] / fund['at']
    
    # G. Other
    fund['tax_rate'] = fund['txp'] / fund['ib']
    fund['tang'] = fund['ppent'] / fund['at']
    fund['depr'] = fund['dp'] / fund['ppent']

    # --- 3. MERGING ---
    print("Linking Datasets (Ticker Match)...")
    df['ticker'] = df['ticker'].astype(str).str.upper()
    fund['tic'] = fund['tic'].astype(str).str.upper()
    
    df['year'] = df['date'].dt.year
    fund['match_year'] = fund['datadate'].dt.year + 1
    
    # Deduplicate
    fund_features = [c for c in fund.columns if c not in ['gvkey', 'datadate', 'fyear', 'tic', 'cusip', 'match_year']]
    fund_dedup = fund.sort_values('datadate').groupby(['tic', 'match_year'])[fund_features].last().reset_index()
    
    df_merged = df.merge(
        fund_dedup,
        left_on=['ticker', 'year'],
        right_on=['tic', 'match_year'],
        how='left',
        suffixes=('', '_comp')
    )
    
    # Scale Level Variables
    scale_cols = ['bm', 'ep', 'sp', 'cp', 'dy', 'gross_prof']
    for c in scale_cols:
        if c in df_merged.columns:
            df_merged[c] = (df_merged[c] * 1000) / df_merged['mkt_cap']

    # Interaction Terms
    df_merged['size_mom'] = df_merged['mkt_cap'] * df_merged['mom12m']
    df_merged['size_value'] = df_merged['mkt_cap'] * df_merged['bm']

    # Drop duplicates
    df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]

    # Final Select
    all_numeric = df_merged.select_dtypes(include=[np.number]).columns.tolist()
    
    # FIX: Added 'mkt_cap' to IDs so it doesn't get selected twice
    ids = ['permno', 'date', 'ret', 'prc', 'shrout', 'vol', 'year', 'match_year', 'fyear', 'mkt_cap']
    predictors = [c for c in all_numeric if c not in ids]
    
    # Construct final list explicitly
    cols = ['permno', 'date', 'ret', 'mkt_cap'] + predictors
    valid_cols = [c for c in cols if c in df_merged.columns]
    
    df_final = df_merged[valid_cols].dropna(subset=['ret', 'mkt_cap', 'mom12m'])
    
    df_final.to_parquet(OUTPUT_FILE)
    
    print("-" * 30)
    print(f"[SUCCESS] GHZ-87 Dataset Saved: {OUTPUT_FILE}")
    print(f"Total Rows: {len(df_final):,}")
    print(f"Total Predictors: {len(predictors)}")
    print("-" * 30)

if __name__ == "__main__":
    build_ghz()