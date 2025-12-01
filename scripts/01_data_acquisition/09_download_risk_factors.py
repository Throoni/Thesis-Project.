import pandas_datareader.data as web
import pandas as pd
import os

DATA_FOLDER = "raw_data"
if not os.path.exists(DATA_FOLDER): os.makedirs(DATA_FOLDER)

def download_ff_factors():
    print("Downloading Fama-French 5 Factors + Momentum...")
    try:
        # Download from Ken French Data Library
        ds_ff5 = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start='1996-01-01')
        df_ff5 = ds_ff5[0]
        
        ds_mom = web.DataReader('F-F_Momentum_Factor', 'famafrench', start='1996-01-01')
        df_mom = ds_mom[0]
        
        df_factors = df_ff5.merge(df_mom, on='Date')
        df_factors = df_factors.reset_index().rename(columns={'Date': 'date'})
        df_factors['date'] = df_factors['date'].dt.to_timestamp() + pd.offsets.MonthEnd(0)
        
        cols = [c for c in df_factors.columns if c != 'date']
        df_factors[cols] = df_factors[cols] / 100.0
        
        save_path = os.path.join(DATA_FOLDER, "ff_factors.parquet")
        df_factors.to_parquet(save_path)
        print(f"[SUCCESS] Saved Fama-French Factors: {len(df_factors)} months.")
        
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")

if __name__ == "__main__":
    download_ff_factors()