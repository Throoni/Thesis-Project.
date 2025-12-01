import wrds
import pandas as pd
import os
import datetime
from config import WRDS_USERNAME

# --- SETTINGS ---
START_DATE = '1996-01-01'
END_DATE = '2024-12-31'  # Get most recent data possible
DATA_FOLDER = "raw_data"

if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

def download_crsp_full():
    print(f"--- Connecting to WRDS as {WRDS_USERNAME} ---")
    try:
        db = wrds.Connection(wrds_username=WRDS_USERNAME)
    except Exception as e:
        print(f"Connection Failed: {e}")
        return

    print(f"\n[START] Downloading CRSP Monthly Stock Data ({START_DATE} to {END_DATE})...")
    print("Querying database (this may take 1-2 minutes)...")

    # Query for Monthly Stock File (msf) and Monthly Stock Event (mse) information
    # We get: 
    # - PRC: Price (negative means average of bid/ask, we will fix this later)
    # - RET: Monthly Return
    # - SHROUT: Shares Outstanding (for Market Cap)
    # - VOL: Volume
    # - BID/ASK: For liquidity measures
    # - SICCD: Industry Code
    
    stock_query = f"""
        SELECT 
            a.permno, a.date, a.prc, a.ret, a.shrout, a.vol, a.bid, a.ask,
            b.siccd, b.ticker
        FROM 
            crsp.msf AS a
        LEFT JOIN
            crsp.msenames AS b
        ON 
            a.permno = b.permno 
            AND a.date >= b.namedt 
            AND a.date <= b.nameendt
        WHERE 
            a.date BETWEEN '{START_DATE}' AND '{END_DATE}'
    """
    
    try:
        df_stocks = db.raw_sql(stock_query)
        
        # Basic Cleaning: Fix negative prices (CRSP convention for bid/ask midpoint)
        df_stocks['prc'] = df_stocks['prc'].abs()
        
        # Basic Feature Engineering: Market Cap
        df_stocks['mkt_cap'] = df_stocks['prc'] * df_stocks['shrout']
        
        row_count = len(df_stocks)
        print(f"[SUCCESS] Downloaded {row_count:,} rows.")
        
        # Save
        save_path = os.path.join(DATA_FOLDER, "crsp_stocks_full.parquet")
        df_stocks.to_parquet(save_path)
        print(f"-> Saved to {save_path}")
        
    except Exception as e:
        print(f"[ERROR] Query failed: {e}")
        print("Possible issue: You might not have CRSP access, or the connection timed out.")

    db.close()

if __name__ == "__main__":
    download_crsp_full()