import wrds
import pandas as pd
import os
from config import WRDS_USERNAME

# 1. Connect to WRDS
print("Connecting to WRDS...")
db = wrds.Connection(wrds_username=WRDS_USERNAME)

# 2. Settings
START_YEAR = 1996
END_YEAR = 2024  # You can adjust this later
DATA_FOLDER = "raw_data"

# Create a folder to store files if it doesn't exist
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# ---------------------------------------------------------
# FUNCTION: DOWNLOAD STOCK DATA (CRSP)
# ---------------------------------------------------------
def download_stocks():
    print(f"\n[1/3] Downloading CRSP Stock Data ({START_YEAR}-{END_YEAR})...")
    
    # We download strictly month-end data for liquidity/momentum calculation
    stock_query = f"""
        SELECT 
            permno, date, prc, ret, shrout, vol, bid, ask
        FROM 
            crsp.msf
        WHERE 
            date BETWEEN '01-01-{START_YEAR}' AND '12-31-{END_YEAR}'
    """
    df_stocks = db.raw_sql(stock_query)
    
    # Save
    save_path = os.path.join(DATA_FOLDER, "crsp_stocks.parquet")
    df_stocks.to_parquet(save_path)
    print(f"   -> Saved {len(df_stocks)} rows to {save_path}")

# ---------------------------------------------------------
# FUNCTION: DOWNLOAD OPTION DATA (OptionMetrics)
# ---------------------------------------------------------
def download_options():
    print(f"\n[2/3] Downloading OptionMetrics Data (Year by Year)...")
    
    # We loop through years because option data is HUGE
    for year in range(START_YEAR, END_YEAR + 1):
        print(f"   Processing {year}...")
        
        # QUERY LOGIC:
        # 1. Calls Only (cp_flag = 'C')
        # 2. Liquidity (open_interest > 0)
        # 3. Maturity: We want options expiring in ~30 days (20 to 40 days) 
        #    to match monthly stock returns.
        
        opt_query = f"""
            SELECT 
                secid, date, exdate, strike_price, best_bid, best_offer, 
                impl_volatility, delta, gamma, vega, theta, open_interest
            FROM 
                optionm.opprcd{year}
            WHERE 
                cp_flag = 'C' 
                AND open_interest > 0
                AND best_bid > 0.125
                AND (exdate - date) BETWEEN 20 AND 40
        """
        try:
            df_opt = db.raw_sql(opt_query)
            if not df_opt.empty:
                save_path = os.path.join(DATA_FOLDER, f"options_{year}.parquet")
                df_opt.to_parquet(save_path)
                print(f"      Saved {len(df_opt)} rows.")
            else:
                print("      No data found (might be a gap year).")
        except Exception as e:
            print(f"      Error downloading {year}: {e}")

# ---------------------------------------------------------
# FUNCTION: DOWNLOAD LINKING TABLE
# ---------------------------------------------------------
def download_link():
    print(f"\n[3/3] Downloading Linking Table...")
    
    # First, try the modern table
    try:
        print("   Trying modern table: wrdsapps.op_crsp_link...")
        link_query = """
            SELECT 
                secid, permno, sdate, edate
            FROM 
                wrdsapps.op_crsp_link
        """
        df_link = db.raw_sql(link_query)
        print(f"   -> Successfully loaded from wrdsapps.op_crsp_link")
    except Exception as e:
        print(f"   -> Modern table failed: {e}")
        print("   Trying fallback: joining optionm.secnmd and crsp.stocknames on cusip...")
        
        try:
            # Fallback: fetch data separately and merge in Python to avoid permission issues
            print("   -> Fetching optionm.secnmd data...")
            optionm_query = """
                SELECT DISTINCT secid, cusip
                FROM optionm.secnmd
                WHERE cusip IS NOT NULL
            """
            df_optionm = db.raw_sql(optionm_query)
            print(f"   -> Fetched {len(df_optionm)} rows from optionm.secnmd")
            
            print("   -> Fetching crsp.stocknames data...")
            crsp_query = """
                SELECT DISTINCT permno, cusip, namedt as sdate, 
                       COALESCE(nameenddt, '9999-12-31'::date) as edate
                FROM crsp.stocknames
                WHERE cusip IS NOT NULL
            """
            df_crsp = db.raw_sql(crsp_query)
            print(f"   -> Fetched {len(df_crsp)} rows from crsp.stocknames")
            
            # Merge on cusip
            print("   -> Merging datasets on cusip...")
            df_link = pd.merge(df_optionm, df_crsp, on='cusip', how='inner')
            df_link = df_link[['secid', 'permno', 'sdate', 'edate']].drop_duplicates()
            print(f"   -> Successfully created link table with {len(df_link)} rows using fallback method")
        except Exception as e2:
            print(f"   -> Fallback method failed: {e2}")
            print("   -> Creating empty link table as placeholder...")
            # Create an empty dataframe with the expected structure
            df_link = pd.DataFrame(columns=['secid', 'permno', 'sdate', 'edate'])
            print("   -> WARNING: Link table is empty. You may need WRDS permissions for optionm.secnmd")
    
    save_path = os.path.join(DATA_FOLDER, "link_table.parquet")
    df_link.to_parquet(save_path)
    print(f"   -> Saved {len(df_link)} rows to {save_path}")

# ---------------------------------------------------------
# EXECUTE
# ---------------------------------------------------------
if __name__ == "__main__":
    # download_stocks()
    download_link()
    # download_options()
    print("\nAll Downloads Complete.")
    db.close()