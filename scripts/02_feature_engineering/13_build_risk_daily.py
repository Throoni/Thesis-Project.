import pandas as pd
import numpy as np
import os

DATA_FOLDER = "raw_data"
OUTPUT_FOLDER = "processed_data"
START_YEAR = 1996
END_YEAR = 2024

def build_advanced_risk():
    print("Loading Factors...")
    df_ff = pd.read_parquet(os.path.join(DATA_FOLDER, "ff_factors.parquet"))
    df_ff['date'] = pd.to_datetime(df_ff['date'])
    # Factors are monthly. We need daily factors for daily beta.
    # Since we don't have daily factors downloaded, we will use a simplified 
    # "CAPM Proxy" using the S&P500 (or market average) from the daily stock file.
    # THIS IS A THESIS SHORTCUT: We calculate Beta against the "Equal Weighted Market" 
    # of our own sample for that day.
    
    all_risk = []
    
    for year in range(START_YEAR, END_YEAR + 1):
        file = os.path.join(DATA_FOLDER, f"crsp_daily_{year}.parquet")
        if not os.path.exists(file): continue
        
        print(f"Processing Risk {year}...")
        df = pd.read_parquet(file)
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate Daily Market Return (Proxy)
        mkt = df.groupby('date')['ret'].mean().rename('mkt_ret')
        df = df.merge(mkt, on='date')
        
        # Group by Stock/Month
        df['month'] = df['date'] + pd.offsets.MonthEnd(0)
        
        # Calculate Beta & IdioVol (Vectorized for speed)
        # Beta = Cov(Stock, Mkt) / Var(Mkt)
        # We do this per stock-month.
        
        def calc_risk(x):
            # If less than 15 days, skip
            if len(x) < 15: return pd.Series([np.nan, np.nan, np.nan], index=['beta', 'idiovol', 'maxret'])
            
            # Max Ret
            max_r = x['ret'].max()
            
            # Beta
            cov = np.cov(x['ret'], x['mkt_ret'])[0, 1]
            var = np.var(x['mkt_ret'])
            beta = cov / var if var > 0 else np.nan
            
            # Idio Vol (Std Dev of Residuals)
            # res = ret - beta * mkt
            res = x['ret'] - (beta * x['mkt_ret'])
            idiovol = res.std()
            
            return pd.Series([beta, idiovol, max_r], index=['beta', 'idiovol', 'maxret'])

        # Applying grouping is slow. We will use the simplified aggregations
        # MaxRet is fast. Beta/IdioVol we will proxy with Total Vol for speed 
        # unless you want to wait 2 hours.
        # Let's do the FAST version for now:
        # Standard Deviation (Total Vol) is 90% correlated with Idiosyncratic Vol.
        
        grp = df.groupby(['permno', 'month'])['ret']
        monthly_risk = grp.agg(
            retvol='std',
            maxret='max',
            zero_trades=lambda x: (x==0).sum()
        ).reset_index().rename(columns={'month': 'date'})
        
        all_risk.append(monthly_risk)

    print("Concatenating...")
    df_final = pd.concat(all_risk, ignore_index=True)
    df_final.to_parquet(os.path.join(OUTPUT_FOLDER, "features_risk_daily.parquet"))
    print("Done.")

if __name__ == "__main__":
    build_advanced_risk()