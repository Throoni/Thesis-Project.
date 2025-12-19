# Stock Return Predictability Using Neural Networks

A Master's Thesis project investigating the predictability of stock returns using ensemble neural networks, incorporating both stock-level and macroeconomic predictors.

## Overview

This project implements a comprehensive machine learning pipeline for predicting monthly stock returns using:
- **Stock-level predictors**: Momentum, liquidity, risk metrics, and fundamental ratios (77+ features)
- **Macroeconomic variables**: VIX, inflation, term spreads, default spreads, and consumer sentiment
- **Advanced modeling**: Ensemble neural networks with proper train/val/test splits and early stopping

The model achieves economically significant out-of-sample R² on monthly returns (2017-2024), with macroeconomic conditions and market structure identified as more predictive than traditional momentum signals.

## Key Methodological Features

✅ **No Data Leakage:** Scaler fits only on training data  
✅ **Proper Validation:** Train/Val/Test split (1996-2013 / 2014-2016 / 2017-2024)  
✅ **Reproducible:** Random seeds set for all stochastic components  
✅ **Realistic Backtest:** Includes transaction costs (10 bps per trade)  
✅ **Industry Standard:** Momentum skips t-1 (Jegadeesh & Titman 1993)

## Project Structure

```
Thesis_Project/
├── scripts/
│   ├── 01_data_acquisition/    # Data download scripts (WRDS, FRED)
│   ├── 02_feature_engineering/ # Feature construction and merging
│   ├── 03_modeling/            # Neural network training
│   ├── 04_evaluation/          # Backtesting and visualization
│   └── utils/                  # Shared utilities (paths, config)
├── raw_data/                   # Raw downloaded data (not in git)
├── processed_data/             # Processed features and datasets (not in git)
├── results/                    # Model outputs and visualizations (not in git)
├── methodology/                # Project documentation
│   ├── 01_code_manifest.md     # Script descriptions
│   ├── 02_project_evolution.md # Development history
│   └── 03_current_status.md    # Current results
├── AUDIT_REPORT.md             # Technical audit and fixes
├── RETRAINING_RESULTS.md       # Results after corrections
├── config.py                   # WRDS credentials (create from example)
└── README.md                   # This file
```

## Prerequisites

- Python 3.8+
- WRDS account with access to CRSP and Compustat
- Required packages: see `requirements.txt`

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Thesis_Project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure WRDS credentials:**
   ```bash
   cp config.example.py config.py
   # Edit config.py with your WRDS username
   ```

## Running the Pipeline

### Step 1: Data Acquisition
```bash
python scripts/01_data_acquisition/03_download_crsp.py
python scripts/01_data_acquisition/05_download_compustat.py
python scripts/01_data_acquisition/11_download_crsp_daily.py
python scripts/01_data_acquisition/19_download_macro.py
```

### Step 2: Feature Engineering
```bash
python scripts/02_feature_engineering/26_build_ghz_predictors.py
python scripts/02_feature_engineering/20_merge_macro.py
```

### Step 3: Model Training
```bash
python scripts/03_modeling/23_train_optimized.py
```

### Step 4: Evaluation
```bash
python scripts/04_evaluation/27_backtest_strategy.py
python scripts/04_evaluation/generate_publication_plots.py
```

## Key Findings

1. **Macroeconomic variables** (inflation, default spreads) are more predictive than traditional momentum signals in recent market regimes
2. **Market structure** (liquidity measures) creates predictable return patterns
3. **Ensemble averaging** improves robustness and out-of-sample performance
4. The model achieves **economically significant** predictive power with Sharpe ratio ~0.5-0.7

## Methodology Highlights

### Data Split Strategy
| Period | Use | Purpose |
|--------|-----|---------|
| 1996-2013 | Training | Model fitting |
| 2014-2016 | Validation | Early stopping |
| 2017-2024 | Test | Final evaluation |

### Feature Categories
- **Momentum** (5): mom1m, mom6m, mom12m, mom36m, chmom
- **Risk** (3): volatility, beta proxy, maxret
- **Liquidity** (4): turnover, dollar volume, spread, zero trades
- **Fundamentals** (30+): valuation, profitability, leverage, growth
- **Macro** (7): VIX, inflation, spreads, sentiment

### Model Architecture
```
Neural Network (x5 Ensemble)
├── Input(77) → Dense(64) + BN + SiLU + Dropout(0.4)
├── Dense(32) + BN + SiLU + Dropout(0.3)
├── Dense(16) + BN + SiLU + Dropout(0.2)
└── Dense(1) → Output
```

## Results

### Out-of-Sample Performance (2017-2024)
- **R²:** ~0.5-0.7% (economically significant)
- **Sharpe (Gross):** ~0.7
- **Sharpe (Net):** ~0.5-0.6

### Top Features
1. Inflation
2. Default Spread
3. CPI
4. Consumer Sentiment
5. VIX

## Data Sources

- **CRSP**: Monthly and daily stock prices, returns, volumes
- **Compustat**: Annual fundamental accounting data
- **FRED**: Macroeconomic variables (VIX, CPI, Treasury yields)

## Documentation

See the `methodology/` folder for detailed documentation:
- **01_code_manifest.md**: Description of all scripts
- **02_project_evolution.md**: History of methodological pivots
- **03_current_status.md**: Current project status and results

## Citation

If you use this code in your research, please cite:

```
[Your Citation Here]
```

## License

See `LICENSE` file for details.

## Acknowledgments

- WRDS (Wharton Research Data Services) for data access
- FRED (Federal Reserve Economic Data) for macroeconomic data
- PyTorch and scikit-learn communities
