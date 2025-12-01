# Code Manifest

This document lists all scripts in the project and their purposes.

## Script Inventory

**Script 01:** `01_download_data.py`  
Downloads OptionMetrics linking table from WRDS, attempting modern table first and falling back to manual CUSIP-based linking.

**Script 02:** `02_diagnose.py`  
Diagnostic script that checks WRDS database access and verifies availability of link tables and option data tables.

**Script 03:** `03_download_crsp.py`  
Downloads full CRSP monthly stock data (prices, returns, shares outstanding) from WRDS for the specified date range.

**Script 04:** `04_inspect_data.py`  
Inspects and displays summary statistics of downloaded CRSP data files including shape, columns, and date ranges.

**Script 05:** `05_download_compustat.py`  
Downloads Compustat annual fundamentals data directly from comp.funda table, processing one year at a time to avoid timeouts.

**Script 06:** `06_build_features.py`  
Processes raw CRSP data to calculate initial technical features including rolling statistics and price-based signals.

**Script 07:** `07_check_stats.py`  
Performs diagnostic checks on processed datasets to verify data quality, coverage, and readiness for machine learning.

**Script 09:** `09_download_risk_factors.py`  
Downloads Fama-French risk factors (Market-RF, SMB, HML, RF) from WRDS for use in risk metric calculations.

**Script 10:** `10_download_compustat_quarterly.py`  
Downloads Compustat quarterly financial data to supplement annual fundamentals for more frequent updates.

**Script 11:** `11_download_crsp_daily.py`  
Downloads CRSP daily stock data year-by-year to calculate high-frequency risk metrics like Beta and Idiosyncratic Volatility.

**Script 12:** `12_build_momentum.py`  
Calculates momentum-based predictors including short-term, medium-term, and long-term return momentum signals.

**Script 13:** `13_build_risk_daily.py`  
Uses daily CRSP data to calculate Beta and Idiosyncratic Volatility by regressing stock returns against market factors.

**Script 14:** `14_build_liquidity.py`  
Calculates liquidity measures including share turnover, bid-ask spreads, and trading volume-based metrics.

**Script 15:** `15_build_fundamentals.py`  
Merges Compustat fundamental data (book equity, market equity, profitability ratios) with CRSP using ticker symbols.

**Script 16:** `16_merge_all.py`  
Merges all feature sets (momentum, risk, liquidity, fundamentals) into a single comprehensive dataset with target variable.

**Script 17:** `17_train_final_model.py`  
Trains a Random Forest model on the final merged dataset to establish baseline predictive performance.

**Script 18:** `18_train_neural_net.py`  
Trains a scikit-learn MLPRegressor neural network with feature importance analysis and learning curve visualization.

**Script 19:** `19_download_macro.py`  
Downloads macroeconomic variables from FRED including VIX, inflation, term spreads, default spreads, and consumer sentiment.

**Script 20:** `20_merge_macro.py`  
Merges macroeconomic predictors with the stock-level dataset, forward-filling to handle date alignment issues.

**Script 21:** `21_train_final_macro_model.py`  
Trains a neural network model incorporating macroeconomic variables and evaluates their contribution to predictions.

**Script 22:** `22_train_pytorch_advanced.py`  
Implements an advanced PyTorch neural network with batch normalization, dropout, and enhanced training procedures.

**Script 23:** `23_train_optimized.py`  
Trains an optimized ensemble neural network with early stopping, best weight restoration, and multiple model averaging for robust predictions.

**Script 24:** `24_expand_features.py`  
Expands the feature set by adding additional fundamental ratios (profitability, leverage, efficiency), seasonality signals (1-5 year lagged returns), and market cap interaction terms to enhance predictive power.

