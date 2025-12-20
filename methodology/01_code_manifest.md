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
Trains an optimized ensemble neural network with early stopping, best weight restoration, gradient clipping, warmup + cosine annealing learning rate schedule, and multiple model averaging for robust predictions.

**Script 24:** `24_expand_features.py`  
Expands the feature set by adding additional fundamental ratios (profitability, leverage, efficiency), seasonality signals (1-5 year lagged returns), and market cap interaction terms to enhance predictive power.

**Script 26:** `26_benchmark_models.py`
Compares neural network performance against baseline models including OLS, Ridge, Lasso, Random Forest, XGBoost, and LightGBM to demonstrate value-add of neural network approach.

**Script 27 (Modeling):** `27_train_advanced.py`
Advanced neural network training script with comprehensive features including gradient clipping, mixed precision training, custom loss functions (MSE, Huber, IC), gradient accumulation, learning rate warmup, and configurable training presets via CLI.

**Script 27 (Evaluation):** `27_backtest_strategy.py`
Backtests the neural network strategy by constructing long-short quintile portfolios, computing Sharpe ratios, and accounting for transaction costs.

**Script 28:** `28_count_predictors.py`
Utility script to count and categorize the predictors used in the final model.

**Script 29:** `29_permutation_importance.py`
Calculates permutation-based feature importance, which is more robust than gradient-based methods, by measuring the drop in R² when each feature is randomly shuffled.

**Script 30:** `30_shap_analysis.py`
Computes SHAP (SHapley Additive exPlanations) values for neural network interpretability. Generates global feature importance, local prediction explanations, and comparison with gradient-based importance.

**Script 25:** `25_hyperparameter_tuning.py`
Uses Optuna for hyperparameter optimization. Tunes learning rate, batch size, hidden dimensions, dropout rates, and activation functions using Tree-structured Parzen Estimator (TPE) with median pruning.

**Script: `test_training_pipeline.py`**
Quick validation script that runs smoke tests on all training pipeline components including imports, configuration presets, loss functions, model architecture, and training utilities.

## Utility Scripts

**`utils/paths.py`**
Provides standardized path utilities for accessing raw data, processed data, and results directories across all scripts.

**`utils/models.py`**
Shared model architectures (AssetPricingNet), early stopping handler, data sanitization utilities, gradient clipping, warmup cosine scheduler, and reproducibility functions used across training and evaluation scripts.

**`utils/losses.py`**
Custom loss functions for financial machine learning including:
- **HuberLoss**: Robust to outliers in financial returns (fat tails)
- **QuantileLoss**: Predict return distributions, not just means
- **AsymmetricLoss**: Different penalties for over/under-prediction
- **SharpeLoss**: Directly optimize risk-adjusted returns (Sharpe ratio)
- **ICLoss**: Optimize Information Coefficient (rank correlation)
- **RankMSELoss**: Weight MSE by prediction rank (focus on extreme predictions)

**`utils/training.py`**
Comprehensive training utilities including:
- **GradientClipper**: Track and clip gradients with statistics
- **WarmupCosineScheduler**: Linear warmup + cosine annealing
- **MixedPrecisionTrainer**: FP16 training for CUDA devices
- **EarlyStopping**: Stop training with best weight restoration
- **ModelCheckpoint**: Save/load model states during training
- **TrainingLogger**: Log metrics to CSV and TensorBoard
- **TrainingConfig**: Dataclass for training configuration

**`utils/training_config.py`**
Configuration management with presets:
- **default**: Balanced configuration for most use cases
- **fast**: Quick experiments with fewer epochs and models
- **robust**: More regularization and longer training
- **huber**: Uses Huber loss for robustness to outliers
- **ic_optimized**: Optimized for Information Coefficient
- **mixed_precision**: FP16 training (CUDA only)
- **large_batch**: Large effective batch size via gradient accumulation

