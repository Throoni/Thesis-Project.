import pandas as pd
import os
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from paths import get_processed_data_path

# --- SETTINGS ---
GHZ_FILE = get_processed_data_path("ghz_predictors_87.parquet")      
RISK_FILE = get_processed_data_path("features_risk_daily.parquet")    
MACRO_FILE = get_processed_data_path("macro_predictors.parquet")      
OUTPUT_FILE = get_processed_data_path("thesis_dataset_macro.parquet")

def merge_final_dataset():
    print("Loading Data...")
    
    # 1. Load Base Predictors (GHZ)
    if not GHZ_FILE.exists():
        print(f"[ERROR] {GHZ_FILE} not found. Run Script 26 first.")
        return
    
    df_ghz = pd.read_parquet(str(GHZ_FILE))
    print(f"Loaded GHZ Source: {len(df_ghz):,} rows")

    # 2. Merge Daily Risk (Beta, MaxRet)
    if RISK_FILE.exists():
        print("Merging Daily Risk Metrics...")
        df_risk = pd.read_parquet(str(RISK_FILE))
        
        # Ensure dates align
        df_risk['date'] = pd.to_datetime(df_risk['date']) + pd.offsets.MonthEnd(0)
        df_ghz['date'] = pd.to_datetime(df_ghz['date']) + pd.offsets.MonthEnd(0)
        
        # Merge and handle potential duplicates immediately
        df_final = df_ghz.merge(df_risk, on=['permno', 'date'], how='left')
    else:
        print("[WARNING] Risk file not found. Skipping.")
        df_final = df_ghz

    # 3. Merge Macro Data
    print("Merging Macro Data...")
    if MACRO_FILE.exists():
        df_macro = pd.read_parquet(str(MACRO_FILE))
        df_macro.index = pd.to_datetime(df_macro.index) + pd.offsets.MonthEnd(0)
        
        df_final = df_final.merge(df_macro, left_on='date', right_index=True, how='left')
        
        # Forward Fill Macro
        df_final = df_final.sort_values('date')
        macro_cols = df_macro.columns.tolist()
        df_final[macro_cols] = df_final[macro_cols].ffill()
    else:
        print("[ERROR] Macro file not found.")
        return

    # 4. Target Check
    if 'future_ret' not in df_final.columns:
        print("Constructing Target Variable (t+1)...")
        df_final['future_ret'] = df_final.groupby('permno')['ret'].shift(-1)
    
    # 5. SAFETY CLEANING (The Fix for "Repetition level mismatch")
    print("Sanitizing Dataset Structure...")
    
    # Drop duplicate columns (e.g. if 'mkt_cap' appeared in both files)
    df_final = df_final.loc[:, ~df_final.columns.duplicated()]
    
    # Reset Index to simple integer index (fixes metadata corruption)
    df_final = df_final.reset_index(drop=True)
    
    # Save
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()  # Delete old corrupted file first
        
    df_final.to_parquet(str(OUTPUT_FILE))
    
    print("-" * 30)
    print(f"FINAL MERGE COMPLETE: {OUTPUT_FILE}")
    print(f"Total Rows: {len(df_final):,}")
    print(f"Total Columns: {len(df_final.columns)}")
    print("-" * 30)

if __name__ == "__main__":
    merge_final_dataset()