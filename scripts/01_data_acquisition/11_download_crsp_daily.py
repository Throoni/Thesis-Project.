import wrds
import pandas as pd
import os
from config import WRDS_USERNAME

# Settings
START_YEAR = 1996
END_YEAR = 2024
DATA_FOLDER = "raw_data"

def download_crsp_daily():
    print(f"--- Connecting to WRDS ---")
    db = wrds.Connection(wrds_username=WRDS_USERNAME)

    print(f"\n[START] Downloading CRSP DAILY ({START_YEAR}-{END_YEAR})...")
    
    # We loop because fetching 30 years of daily data at once will crash memory
    for year in range(START_YEAR, END_YEAR + 1):
        print(f"   Processing {year}...")
        
        # We only need: 
        # PERMNO (ID), DATE, RET (Return), VOL (Volume), PRC (Price)
        # Used for: Beta, Idiosyncratic Volatility, Max Daily Return, Illiquidity
        query = f"""
            SELECT 
                permno, date, prc, ret, vol
            FROM 
                crsp.dsf
            WHERE 
                date BETWEEN '{year}-01-01' AND '{year}-12-31'
        """
        
        try:
            df = db.raw_sql(query)
            
            if not df.empty:
                # Optimization: Downcast floats to save space (float64 -> float32)
                df['ret'] = pd.to_numeric(df['ret'], errors='coerce').astype('float32')
                df['prc'] = pd.to_numeric(df['prc'], errors='coerce').astype('float32')
                
                save_path = os.path.join(DATA_FOLDER, f"crsp_daily_{year}.parquet")
                df.to_parquet(save_path)
                print(f"      Saved {len(df):,} rows to {save_path}")
        except Exception as e:
            print(f"      [ERROR] {year}: {e}")

    print("All Daily Data Downloaded.")
    db.close()

if __name__ == "__main__":
    download_crsp_daily()