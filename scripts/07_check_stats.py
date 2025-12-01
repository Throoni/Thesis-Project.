import pandas as pd
import os

# --- SETTINGS ---
FILE_PATH = "processed_data/stock_predictors.parquet"

def check_stats():
    print("-" * 50)
    print(f"DIAGNOSTIC: Checking {FILE_PATH}...")
    print("-" * 50)

    if not os.path.exists(FILE_PATH):
        print(f"[ERROR] File not found: {FILE_PATH}")
        print("Did Step 6 (build_features) finish successfully?")
        return

    # Load the data
    df = pd.read_parquet(FILE_PATH)
    
    # 1. Basic Counts
    num_obs = len(df)
    num_stocks = df['permno'].nunique()
    
    # 2. Timeframe
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    num_months = df['date'].nunique()
    
    # 3. Predictor Coverage
    # We define predictors as anything that isn't an ID or Price
    id_cols = ['permno', 'date', 'ticker', 'ret', 'prc', 'shrout']
    predictors = [c for c in df.columns if c not in id_cols]
    
    print(f"Total Observations:      {num_obs:,}")
    print(f"Unique Stocks:           {num_stocks:,}")
    print(f"Time Period:             {min_date} to {max_date} ({num_months} months)")
    print(f"Avg Stocks per Month:    {int(num_obs / num_months):,}")
    
    print("-" * 50)
    print(f"PREDICTORS AVAILABLE ({len(predictors)}):")
    for p in predictors:
        # Calculate how much data is missing for each predictor
        missing = df[p].isna().sum()
        missing_pct = (missing / num_obs) * 100
        print(f"  - {p:<15} (Missing: {missing_pct:.1f}%)")
        
    print("-" * 50)
    
    # 4. Thesis Viability Check
    if num_obs > 1_000_000 and len(predictors) >= 5:
        print("✅ STATUS: READY FOR MACHINE LEARNING.")
        print("   You have enough data to replicate Gu, Kelly, & Xiu (2020).")
    else:
        print("⚠️ STATUS: DATASET MIGHT BE TOO SMALL.")
        print("   Check if the download scripts completed correctly.")

if __name__ == "__main__":
    check_stats()