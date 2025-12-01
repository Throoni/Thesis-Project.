import wrds
import pandas as pd
import os
from config import WRDS_USERNAME

START_DATE = '1996-01-01'
END_DATE = '2024-12-31'
DATA_FOLDER = "raw_data"

def download_compustat_quarterly():
    print(f"--- Connecting to WRDS as {WRDS_USERNAME} ---")
    try:
        db = wrds.Connection(wrds_username=WRDS_USERNAME)
    except Exception as e:
        print(f"Connection Failed: {e}")
        return

    print(f"\n[START] Downloading Compustat QUARTERLY ({START_DATE}-{END_DATE})...")
    
    # We need specific quarterly columns for the 94 predictors
    # cogsq: Cost of Goods Sold
    # saleq: Sales
    # atq: Total Assets
    # ltq: Liabilities
    # ibq: Income Before Extraordinary Items
    # niq: Net Income
    query = f"""
        SELECT 
            gvkey, datadate, fyearq, fqtr,
            atq, ltq, seqq, pstkq, txditcq, ibq, saleq, cogsq, niq,
            cusip, tic
        FROM 
            comp.fundq
        WHERE 
            indfmt='INDL'
            AND datafmt='STD'
            AND popsrc='D'
            AND consol='C'
            AND datadate BETWEEN '{START_DATE}' AND '{END_DATE}'
    """
    
    try:
        df_fund = db.raw_sql(query)
        print(f"[SUCCESS] Downloaded {len(df_fund):,} rows.")
        
        # Basic clean
        df_fund['cusip_8'] = df_fund['cusip'].str.slice(0, 8)
        
        save_path = os.path.join(DATA_FOLDER, "compustat_quarterly.parquet")
        df_fund.to_parquet(save_path)
        print(f"-> Saved to {save_path}")
        
    except Exception as e:
        print(f"[ERROR] Query failed: {e}")

    db.close()

if __name__ == "__main__":
    download_compustat_quarterly()