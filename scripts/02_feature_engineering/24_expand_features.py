import pandas as pd
import numpy as np
import os

# --- SETTINGS ---
STOCK_FILE = "processed_data/stock_predictors.parquet"
COMP_FILE = "raw_data/compustat_annual.parquet"
OUTPUT_FILE = "processed_data/stock_predictors_expanded.parquet"

def expand_features_mega():
    print("Loading existing dataset...")
    df = pd.read_parquet(STOCK_FILE)
    df['date'] = pd.to_datetime(df['date'])  # Ensure date is datetime
    
    print("Loading EXTENDED Fundamentals...")
    df_fund = pd.read_parquet(COMP_FILE)
    df_fund['datadate'] = pd.to_datetime(df_fund['datadate'])
    
    # Handle zeros/NaNs to prevent division errors
    for col in df_fund.columns:
        if df_fund[col].dtype in ['float64', 'float32']:
            df_fund[col] = df_fund[col].replace(0, np.nan)

    # --- 1. VALUATION & PROFITABILITY (10 Predictors) ---
    print("Calculating Valuation & Profitability...")
    # Operating Profitability: (Sales - COGS) / Assets
    df_fund['oper_prof'] = (df_fund['sale'] - df_fund['cogs']) / df_fund['at']
    # Gross Profitability: (Sales - COGS)
    df_fund['gross_prof'] = df_fund['sale'] - df_fund['cogs']
    # Return on Equity
    df_fund['roe'] = df_fund['ib'] / df_fund['seq']
    # Return on Assets
    df_fund['roa'] = df_fund['ib'] / df_fund['at']
    # Cash Flow to Assets
    df_fund['cf_assets'] = df_fund['oancf'] / df_fund['at']
    # Dividend Yield (Dividends / Assets as proxy, normalized by price later)
    df_fund['div_yield_proxy'] = df_fund['dv']
    # R&D Intensity
    df_fund['rd_sales'] = df_fund['xrd'] / df_fund['sale']
    df_fund['rd_assets'] = df_fund['xrd'] / df_fund['at']
    # Tax Rate
    df_fund['tax_rate'] = df_fund['txp'] / df_fund['ib']
    # Asset Turnover
    df_fund['asset_turnover'] = df_fund['sale'] / df_fund['at']

    # --- 2. FINANCIAL HEALTH & LEVERAGE (8 Predictors) ---
    print("Calculating Financial Health...")
    # Current Ratio
    df_fund['cur_ratio'] = df_fund['act'] / df_fund['lct']
    # Quick Ratio: (Current Assets - Inventory) / Current Liab
    df_fund['quick_ratio'] = (df_fund['act'] - df_fund['invt']) / df_fund['lct']
    # Cash Ratio
    df_fund['cash_ratio'] = df_fund['che'] / df_fund['lct']
    # Leverage (Debt / Assets)
    df_fund['debt_assets'] = (df_fund['dltt'] + df_fund['dlc']) / df_fund['at']
    # Debt to Equity
    df_fund['debt_equity'] = (df_fund['dltt'] + df_fund['dlc']) / df_fund['seq']
    # Inventory Turnover
    df_fund['inv_turnover'] = df_fund['cogs'] / df_fund['invt']
    # Receivables Turnover
    df_fund['rec_turnover'] = df_fund['sale'] / df_fund['rect']
    # Tangibility (PPE / Assets)
    df_fund['tangibility'] = df_fund['ppent'] / df_fund['at']

    # --- 3. GROWTH SIGNALS (5 Predictors) ---
    print("Calculating Growth...")
    # Sort by firm and date to calculate changes
    df_fund = df_fund.sort_values(['gvkey', 'datadate'])
    
    # Asset Growth
    df_fund['asset_growth'] = df_fund.groupby('gvkey')['at'].pct_change()
    # Sales Growth
    df_fund['sales_growth'] = df_fund.groupby('gvkey')['sale'].pct_change()
    # Earnings Growth
    df_fund['earnings_growth'] = df_fund.groupby('gvkey')['ib'].pct_change()
    # Debt Growth
    df_fund['debt_growth'] = df_fund.groupby('gvkey')['dltt'].pct_change()
    # R&D Growth
    df_fund['rd_growth'] = df_fund.groupby('gvkey')['xrd'].pct_change()

    # --- MERGE WITH CRSP ---
    print("Merging...")
    df_fund['tic'] = df_fund['tic'].astype(str).str.upper()
    df_fund['match_year'] = df_fund['datadate'].dt.year + 1
    
    # Select the new columns
    new_cols = ['oper_prof', 'gross_prof', 'roe', 'roa', 'cf_assets', 'div_yield_proxy',
                'rd_sales', 'rd_assets', 'tax_rate', 'asset_turnover',
                'cur_ratio', 'quick_ratio', 'cash_ratio', 'debt_assets', 'debt_equity',
                'inv_turnover', 'rec_turnover', 'tangibility',
                'asset_growth', 'sales_growth', 'earnings_growth', 'debt_growth', 'rd_growth']
    
    # Deduplicate
    df_fund_unique = df_fund.sort_values('datadate').groupby(['tic', 'match_year'])[new_cols].last().reset_index()
    
    df['year'] = df['date'].dt.year
    df = df.merge(
        df_fund_unique, 
        left_on=['ticker', 'year'], 
        right_on=['tic', 'match_year'], 
        how='left'
    ).drop(columns=['tic', 'match_year'], errors='ignore')

    # --- 4. NEW PRICE PREDICTORS (5 Predictors) ---
    # Creating "Beta Squared" and "Idiosyncratic Volatility" proxies
    # (Since we didn't save daily residuals, we approximate using monthly)
    print("Calculating Advanced Price Signals...")
    
    # Beta Proxy: Covariance with Market (approximated by correlation * vol ratio)
    # This is a simplified rolling beta
    # We'll use Volatility as a proxy for risk for now, but add:
    
    # Seasonality (Heston & Sadka)
    grouped = df.groupby('permno')['ret']
    df['season_1y'] = grouped.shift(12)
    df['season_2y'] = grouped.shift(24)
    df['season_3y'] = grouped.shift(36)
    df['season_4y'] = grouped.shift(48)
    df['season_5y'] = grouped.shift(60)
    
    # Size * Momentum Interaction (Classic Gu et al predictor)
    df['size_mom_interact'] = df['mkt_cap'] * df['mom12m']
    
    # Save
    df.to_parquet(OUTPUT_FILE)
    
    total_predictors = len(new_cols) + 6  # +6 for price signals (5 seasonality + 1 interaction)
    print("-" * 30)
    print(f"[SUCCESS] Added {total_predictors} NEW predictors.")
    print(f"Total Columns: {len(df.columns)}")
    print("-" * 30)

if __name__ == "__main__":
    expand_features_mega()
