import pandas_datareader.data as web
import pandas as pd
import os

DATA_FOLDER = "raw_data"
OUTPUT_FOLDER = "processed_data"

def download_macro():
    print("Downloading Macro Data from FRED...")
    
    # 8 Key Predictors from Welch & Goyal (2008)
    # T10Y2Y: Term Spread (10y - 2y Treasury)
    # AAA10Y: Default Spread (Corporate - Treasury)
    # DTB3:   Risk Free Rate (3-month T-Bill)
    # VIXCLS: VIX Index (Volatility)
    # CPIAUCSL: Inflation (CPI)
    # UMCSENT: Consumer Sentiment
    
    indicators = {
        'T10Y2Y': 'term_spread',
        'BAMLC0A0CM': 'default_spread', # ICE BofA US Corp Master Option-Adjusted Spread
        'DTB3': 'risk_free',
        'VIXCLS': 'vix',
        'CPIAUCSL': 'cpi',
        'UMCSENT': 'sentiment'
    }
    
    try:
        df = web.DataReader(list(indicators.keys()), 'fred', start='1996-01-01', end='2024-12-31')
        df = df.rename(columns=indicators)
        
        # Resample to Monthly (End of Month) to match Stocks
        df_monthly = df.resample('ME').last()
        
        # Calculate Inflation Rate (YoY Change in CPI)
        df_monthly['inflation'] = df_monthly['cpi'].pct_change(12)
        
        # Fill VIX (started in 1990s, but some gaps might exist)
        df_monthly = df_monthly.ffill()
        
        # Save
        save_path = os.path.join(OUTPUT_FOLDER, "macro_predictors.parquet")
        df_monthly.to_parquet(save_path)
        print(f"[SUCCESS] Macro Data Saved: {len(df_monthly)} months.")
        print(df_monthly.tail())
        
    except Exception as e:
        print(f"[ERROR] Macro Download Failed: {e}")

if __name__ == "__main__":
    download_macro()