import wrds
import pandas as pd
import os
from config import WRDS_USERNAME

# --- SETTINGS ---
START_DATE = '1996-01-01'
END_DATE = '2024-12-31'
DATA_FOLDER = "raw_data"

def download_compustat():
    print(f"--- Connecting to WRDS as {WRDS_USERNAME} ---")
    try:
        db = wrds.Connection(wrds_username=WRDS_USERNAME)
    except Exception as e:
        print(f"Connection Failed: {e}")
        return

    print(f"\n[START] Downloading EXTENDED Compustat Fundamentals ({START_DATE} to {END_DATE})...")
    
    all_years = []
    
    for year in range(1996, 2025):
        print(f"   Processing {year}...")
        
        # EXTENDED QUERY: Includes cogs, inventory, R&D, debt, etc.
        query = f"""
            SELECT 
                gvkey, datadate, fyear,
                at, lt, seq, pstk, txditc, ib, sale,
                che, rect, invt, act, lct, dltt, dlc, txp, dp, xrd, dv, cogs, ppent, oancf,
                cusip, tic
            FROM 
                comp.funda
            WHERE 
                indfmt='INDL'
                AND datafmt='STD'
                AND popsrc='D'
                AND consol='C'
                AND datadate BETWEEN '{year}-01-01' AND '{year}-12-31'
        """
        
        try:
            df_year = db.raw_sql(query)
            if not df_year.empty:
                all_years.append(df_year)
        except Exception as e:
            print(f"      [ERROR] {year}: {e}")

    if all_years:
        print("Concatenating data...")
        df_final = pd.concat(all_years, ignore_index=True)
        df_final['datadate'] = pd.to_datetime(df_final['datadate'])
        
        # Ensure raw_data folder exists relative to where we run the script
        if not os.path.exists(DATA_FOLDER):
            os.makedirs(DATA_FOLDER)
            
        save_path = os.path.join(DATA_FOLDER, "compustat_annual.parquet")
        df_final.to_parquet(save_path)
        print("-" * 30)
        print(f"[SUCCESS] Saved {len(df_final):,} rows.")
        print("Columns downloaded:", df_final.columns.tolist())
        print("-" * 30)
    else:
        print("[FAILURE] No data downloaded.")

    db.close()

if __name__ == "__main__":
    download_compustat()
