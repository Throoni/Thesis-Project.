# Stock Return Predictability Using Neural Networks

A Master's Thesis project investigating the predictability of stock returns using ensemble neural networks, incorporating both stock-level and macroeconomic predictors.

## Overview

This project implements a comprehensive machine learning pipeline for predicting monthly stock returns using:
- **Stock-level predictors**: Momentum, liquidity, risk metrics, and fundamental ratios
- **Macroeconomic variables**: VIX, inflation, term spreads, default spreads, and consumer sentiment
- **Advanced modeling**: Ensemble neural networks with early stopping and best weight restoration

The model achieves economically significant out-of-sample R² of ~1.00% on monthly returns (1996-2024), identifying macroeconomic conditions and market structure as more predictive than traditional momentum signals in the post-2016 regime.

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
│   ├── 01_code_manifest.md
│   ├── 02_project_evolution.md
│   └── 03_current_status.md
├── config.py                   # WRDS credentials (create from config.example.py)
└── README.md                   # This file
```

## Prerequisites

- Python 3.8+
- WRDS account with access to:
  - CRSP (Center for Research in Security Prices)
  - Compustat
- Required Python packages (see `requirements.txt` or install via pip):
  - `pandas`, `numpy`
  - `torch` (PyTorch)
  - `scikit-learn`
  - `matplotlib`, `seaborn`
  - `wrds` (WRDS Python API)
  - `pandas-datareader` (for FRED data)

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

4. **Run the pipeline:**
   ```bash
   # Data acquisition
   python scripts/01_data_acquisition/03_download_crsp.py
   python scripts/01_data_acquisition/05_download_compustat.py
   python scripts/01_data_acquisition/19_download_macro.py
   
   # Feature engineering
   python scripts/02_feature_engineering/26_build_ghz_predictors.py
   python scripts/02_feature_engineering/20_merge_macro.py
   
   # Model training
   python scripts/03_modeling/23_train_optimized.py
   
   # Evaluation and visualization
   python scripts/04_evaluation/generate_publication_plots.py
   ```

## Key Findings

1. **Macroeconomic variables** (inflation, default spreads) are more predictive than traditional momentum signals in recent market regimes
2. **Market structure** (liquidity measures) creates predictable return patterns
3. **Ensemble averaging** improves robustness and out-of-sample performance
4. The model achieves **economically significant** predictive power despite modest R² (~1.00%)

## Methodology

See the `methodology/` folder for detailed documentation:
- **01_code_manifest.md**: Description of all scripts
- **02_project_evolution.md**: History of methodological pivots and challenges
- **03_current_status.md**: Current project status and results

## Data Sources

- **CRSP**: Monthly and daily stock prices, returns, trading volumes
- **Compustat**: Annual and quarterly fundamental accounting data
- **FRED**: Macroeconomic variables (VIX, CPI, Treasury yields, etc.)

## Results

Publication-quality figures are generated in `results/`:
- `publication_feature_importance.png`: Top 15 features by importance
- `publication_learning_curve.png`: Training dynamics with early stopping
- `publication_wealth_curve.png`: Out-of-sample strategy performance

## Citation

If you use this code in your research, please cite:

```
[Your Citation Here]
```

## License

See `LICENSE` file for details.

## Contact

[Your Contact Information]

## Acknowledgments

- WRDS (Wharton Research Data Services) for data access
- FRED (Federal Reserve Economic Data) for macroeconomic data
- PyTorch and scikit-learn communities

