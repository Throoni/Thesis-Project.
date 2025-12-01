import pandas as pd
import numpy as np
import os

# --- SETTINGS ---
CRSP_FILE = "raw_data/crsp_stocks_full.parquet"
COMP_ANNUAL = "raw_data/compustat_annual.parquet"
COMP_QUARTER = "raw_data/compustat_quarterly.parquet"
OUTPUT_FOLDER = "processed_data"

def build_fundamentals():
    print("Loading CRSP (for linking)...")
    df_crsp = pd.read_parquet(CRSP_FILE)
    df_crsp['date'] = pd.to_datetime(df_crsp['date'])
    df_crsp['ticker'] = df_crsp['ticker'].astype(str).str.upper()
    df_crsp['year'] = df_crsp['date'].dt.year
    
    # We only need the keys from CRSP to link
    df_link = df_crsp[['permno', 'date', 'ticker', 'year', 'prc', 'shrout']].copy()
    df_link['prc'] = df_link['prc'].abs()
    df_link['mkt_cap'] = df_link['prc'] * df_link['shrout'] # in Thousands
    
    # --- PART A: ANNUAL (Book-to-Market, Earnings-Price) ---
    if os.path.exists(COMP_ANNUAL):
        print("Processing Annual Fundamentals...")
        df_ann = pd.read_parquet(COMP_ANNUAL)
        df_ann['datadate'] = pd.to_datetime(df_ann['datadate'])
        df_ann['tic'] = df_ann['tic'].astype(str).str.upper()
        
        # Create Matching Year (Fiscal Data available t+1)
        df_ann['match_year'] = df_ann['datadate'].dt.year + 1
        
        # Calc Book Equity
        df_ann['seq'] = df_ann['seq'].fillna(0)
        df_ann['txditc'] = df_ann['txditc'].fillna(0)
        df_ann['pstk'] = df_ann['pstk'].fillna(0)
        df_ann['be'] = df_ann['seq'] + df_ann['txditc'] - df_ann['pstk']
        
        # Deduplicate (Take last report per ticker/year)
        df_ann_unique = df_ann.sort_values('datadate').groupby(['tic', 'match_year']).last().reset_index()
        
        # Merge
        df_merged = pd.merge(
            df_link, 
            df_ann_unique[['tic', 'match_year', 'be', 'ib', 'sale']], 
            left_on=['ticker', 'year'], 
            right_on=['tic', 'match_year'], 
            how='left'
        )
        
        # Calc Ratios
        # Compustat in Millions, CRSP in Thousands -> * 1000
        df_merged['bm'] = (df_merged['be'] * 1000) / df_merged['mkt_cap']
        df_merged['ep'] = (df_merged['ib'] * 1000) / df_merged['mkt_cap']
        df_merged['sp'] = (df_merged['sale'] * 1000) / df_merged['mkt_cap']
        
        # Keep just the signals
        df_annual_signals = df_merged[['permno', 'date', 'bm', 'ep', 'sp']]
    else:
        print("Warning: Annual Compustat file not found.")
        df_annual_signals = pd.DataFrame()

    # --- PART B: QUARTERLY (Asset Growth, ROE) ---
    if os.path.exists(COMP_QUARTER):
        print("Processing Quarterly Fundamentals...")
        df_q = pd.read_parquet(COMP_QUARTER)
        df_q['datadate'] = pd.to_datetime(df_q['datadate'])
        df_q['tic'] = df_q['tic'].astype(str).str.upper()
        
        # Sort for growth calc
        df_q = df_q.sort_values(['tic', 'datadate'])
        
        # Asset Growth (YoY change in Total Assets)
        # Shift 4 quarters back
        df_q['atq_lag4'] = df_q.groupby('tic')['atq'].shift(4)
        df_q['asset_growth'] = (df_q['atq'] - df_q['atq_lag4']) / df_q['atq_lag4']
        
        # ROE (Net Income / Shareholder Equity)
        df_q['roe'] = df_q['niq'] / df_q['seqq']
        
        # To merge Quarterly data to Monthly prices, we "Forward Fill"
        # We assume quarterly data is valid for 3 months after publication
        # Strategy: Create a 'valid_month' key? 
        # Easier Strategy: Merge on Ticker/Year/Quarter? Hard to align.
        # Thesis Hack: Just take the most recent available quarterly data for that year
        # We will skip complex quarterly merging for now to ensure script stability.
        # We will just save the Annual signals which are the most important.
        pass 

    # Save
    if not df_annual_signals.empty:
        save_path = os.path.join(OUTPUT_FOLDER, "features_fundamental.parquet")
        df_annual_signals.to_parquet(save_path)
        print(f"[SUCCESS] Fundamental features saved: {len(df_annual_signals):,} rows.")

if __name__ == "__main__":
    build_fundamentals()