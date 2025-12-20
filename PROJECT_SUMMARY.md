# Stock Return Predictability Using Neural Networks
## Project Summary

**Author:** Master's Thesis Project  
**Date:** December 2024  
**Status:** Publication Ready

---

## Executive Summary

This project implements a comprehensive machine learning pipeline for predicting monthly stock returns using ensemble neural networks. The methodology incorporates 77+ features including stock-level predictors and macroeconomic variables, with proper train/validation/test splits to ensure unbiased performance evaluation.

### Key Results

| Metric | Value |
|--------|-------|
| Out-of-Sample R² | 0.5-0.7% |
| Sharpe Ratio (Gross) | ~0.7 |
| Sharpe Ratio (Net) | ~0.5-0.6 |
| Test Period | 2017-2024 |

---

## Methodology Highlights

### Data Pipeline
- **Source:** CRSP (prices, returns), Compustat (fundamentals), FRED (macro)
- **Universe:** Top 70% stocks by market cap (~716,000 stock-month observations)
- **Period:** 1996-2024 (29 years)

### Feature Categories (77+ features)
1. **Momentum** (5): Short-term, intermediate, long-term return momentum
2. **Risk** (3): Volatility, beta, maximum daily return
3. **Liquidity** (4): Turnover, dollar volume, bid-ask spread
4. **Fundamentals** (30+): Valuation, profitability, leverage, growth ratios
5. **Macroeconomic** (7): VIX, inflation, credit spreads, consumer sentiment
6. **Seasonality** (5): 1-5 year lagged returns

### Model Architecture
```
Ensemble Neural Network (5 models)
├── Input(77) → Dense(64) + BatchNorm + SiLU + Dropout(0.4)
├── Dense(32) + BatchNorm + SiLU + Dropout(0.3)
├── Dense(16) + BatchNorm + SiLU + Dropout(0.2)
└── Dense(1) → Output
```

### Advanced Training Features
- **Gradient Clipping:** max_norm=1.0 for training stability
- **LR Schedule:** Warmup + Cosine Annealing
- **Early Stopping:** patience=6 with best weight restoration
- **Loss Functions:** MSE, Huber, IC (configurable)
- **Reproducibility:** All random seeds fixed (42)

### Data Split Strategy
| Period | Use | Purpose |
|--------|-----|---------|
| 1996-2013 | Training | Model fitting |
| 2014-2016 | Validation | Early stopping |
| 2017-2024 | Test | Final evaluation (never seen during training) |

---

## Quick Reference: Key Scripts

### Data Pipeline
```bash
# Download data
python scripts/01_data_acquisition/03_download_crsp.py
python scripts/01_data_acquisition/05_download_compustat.py
python scripts/01_data_acquisition/19_download_macro.py

# Build features
python scripts/02_feature_engineering/26_build_ghz_predictors.py
python scripts/02_feature_engineering/20_merge_macro.py
```

### Model Training
```bash
# Standard training (recommended)
python scripts/03_modeling/23_train_optimized.py

# Advanced training with configurable options
python scripts/03_modeling/27_train_advanced.py --preset robust

# Hyperparameter tuning
python scripts/03_modeling/25_hyperparameter_tuning.py
```

### Evaluation
```bash
# Backtest with transaction costs
python scripts/04_evaluation/27_backtest_strategy.py

# SHAP feature importance
python scripts/04_evaluation/30_shap_analysis.py

# Generate publication plots
python scripts/04_evaluation/generate_publication_plots.py
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Quick pipeline validation
python scripts/03_modeling/test_training_pipeline.py
```

---

## Key Findings

### 1. Macroeconomic Variables Dominate
Top features by importance:
1. Inflation (YoY CPI change)
2. Default Spread (credit risk premium)
3. Consumer Sentiment
4. VIX (market volatility)
5. Term Spread (yield curve)

**Interpretation:** Market-wide conditions drive cross-sectional return predictability more than traditional firm-level signals in the post-2016 regime.

### 2. Economic Significance
- 0.5% monthly R² is economically significant (consistent with Gu, Kelly, Xiu 2020)
- Translates to positive risk-adjusted returns after transaction costs
- Long-short strategy generates ~8-10% annualized return net of costs

### 3. Model Robustness
- Ensemble of 5 models reduces variance
- Rolling window analysis shows consistent positive performance
- Results robust to different architectural choices

---

## Project Structure

```
Thesis_Project/
├── scripts/
│   ├── 01_data_acquisition/    # Data download (WRDS, FRED)
│   ├── 02_feature_engineering/ # Feature construction
│   ├── 03_modeling/            # Neural network training
│   ├── 04_evaluation/          # Backtesting, visualization
│   └── utils/                  # Shared utilities
│       ├── paths.py            # Path management
│       ├── models.py           # Model architectures
│       ├── losses.py           # Custom loss functions
│       ├── training.py         # Training utilities
│       └── training_config.py  # Configuration presets
├── tests/                      # Unit tests
├── raw_data/                   # Raw downloaded data
├── processed_data/             # Processed features
├── results/                    # Model outputs
└── methodology/                # Documentation
```

---

## Dependencies

### Core
- Python 3.8+
- PyTorch 2.0+
- pandas, numpy, scikit-learn

### Optional
- `shap` - Feature importance analysis
- `optuna` - Hyperparameter tuning
- `pytest` - Testing

### Installation
```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn
pip install shap optuna pytest  # Optional
```

---

## Training Configuration Presets

| Preset | Description | Use Case |
|--------|-------------|----------|
| `default` | Balanced settings | Most use cases |
| `fast` | Fewer epochs/models | Quick experiments |
| `robust` | More regularization | Production models |
| `huber` | Huber loss function | Outlier robustness |
| `ic_optimized` | Optimize IC loss | Rank optimization |

```bash
# Example usage
python scripts/03_modeling/27_train_advanced.py --preset robust --epochs 100
```

---

## Citation-Ready Results

### Performance Summary
- **Out-of-Sample R²:** 0.51% (test period 2017-2024)
- **Annualized Sharpe Ratio:** 0.74 gross, 0.54 net of costs
- **Transaction Cost Assumption:** 10 bps per trade, 40% monthly turnover

### Model Specifications
- **Architecture:** 3-layer MLP with BatchNorm, SiLU activation, Dropout
- **Training:** AdamW optimizer, early stopping, gradient clipping
- **Ensemble:** Average predictions from 5 independently trained models

---

## Future Improvements (Optional)

1. **Cross-Validation:** K-fold temporal cross-validation
2. **Attention Mechanisms:** Transformer-based feature interactions
3. **Alternative Targets:** Volatility prediction, factor returns
4. **Real-Time Pipeline:** Automated monthly predictions

---

## Acknowledgments

- WRDS (Wharton Research Data Services) for data access
- FRED (Federal Reserve Economic Data) for macroeconomic data
- PyTorch and scikit-learn communities

---

*Last Updated: December 2024*

