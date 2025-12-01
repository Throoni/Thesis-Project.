# Project Evolution

This document chronicles the key methodological pivots and challenges encountered during the development of this Master's Thesis project.

## The OptionMetrics Blocker

**Original Intent:** The project initially aimed to predict **Option Returns** using the full IvyDB US database from OptionMetrics. This would have allowed for sophisticated analysis of options pricing and volatility prediction.

**Challenge:** Despite having WRDS access, the research team lacked subscription access to the complete OptionMetrics database. Critical tables such as `wrdsapps.link_crsp_optionm` and `optionm.secnmd` were inaccessible due to insufficient permissions, blocking the original research design.

**Impact:** This forced a fundamental pivot in research direction, moving from options-based research to equity-based research.

---

## The Pivot: Stock Return Predictability

**New Direction:** Following the OptionMetrics blocker, the project pivoted to **Stock Return Predictability** using publicly available and accessible datasets.

**Data Sources Secured:**
- **CRSP (Center for Research in Security Prices):** Monthly and daily stock price data, returns, trading volumes, and market microstructure variables
- **Compustat:** Annual and quarterly fundamental accounting data including balance sheet and income statement items

**Rationale:** Both CRSP and Compustat are standard datasets in finance research and were fully accessible through the WRDS subscription, providing a robust foundation for return prediction models.

---

## The Link Table Workaround

**Challenge:** The standard approach for merging CRSP and Compustat data uses the **CCM (CRSP/Compustat Merged) Link Table** (`crsp.ccmxpf_linktable`), which provides official PERMNO-GVKEY mappings maintained by WRDS.

**Problem:** Access to the CCM Link Table was restricted, preventing the use of the standard linking methodology.

**Solution:** Developed custom linking logic that merges datasets using **Ticker symbols** (`tic` field) as the primary key. This approach:
- Extracts ticker symbols from both CRSP and Compustat datasets
- Performs fuzzy matching and date alignment to handle ticker changes over time
- Creates a manual linking table that approximates the functionality of the official CCM table

**Limitations:** Ticker-based linking is less robust than PERMNO-GVKEY linking, as tickers can change due to corporate actions, but this workaround enabled the project to proceed.

---

## The Neural Net Crash

**Initial Implementation:** Early neural network models (Scripts 17-18) were implemented using scikit-learn's MLPRegressor, but encountered catastrophic failures.

**Symptoms:**
- Models produced **negative R² values** (worse than predicting the mean)
- Training loss contained **Infinity (Inf) and NaN values**
- Models failed to converge or produced nonsensical predictions

**Root Cause:** The dataset contained extreme outliers, missing values, and improperly scaled features that caused numerical instability in the neural network optimization process.

**The Nuclear Sanitization Fix:** Implemented a comprehensive data cleaning pipeline that:
1. **Removes infinite values:** Replaces Inf and -Inf with NaN
2. **Handles extreme outliers:** Winsorizes features at the 1st and 99th percentiles
3. **Imputes missing values:** Uses forward-fill and median imputation strategies
4. **Standardizes features:** Applies StandardScaler to ensure all features are on comparable scales
5. **Drops invalid rows:** Removes any remaining rows with NaN or Inf values after cleaning

This "nuclear sanitization" block became a critical preprocessing step in all subsequent modeling efforts (Scripts 21-23).

---

## The Macro Upgrade

**Motivation:** Initial models using only stock-level predictors (momentum, liquidity, fundamentals) showed limited predictive power. The literature (Welch & Goyal, 2008) suggests that macroeconomic variables are crucial for return prediction.

**Implementation:** 
- **Script 19:** Downloads macroeconomic variables from FRED (Federal Reserve Economic Data):
  - **VIX:** Volatility index (market fear gauge)
  - **Inflation (CPI):** Consumer Price Index
  - **Term Spread:** 10-year minus 2-year Treasury yield
  - **Default Spread:** Corporate bond yield minus Treasury yield
  - **Risk-Free Rate:** 3-month T-Bill rate
  - **Consumer Sentiment:** University of Michigan Consumer Sentiment Index

- **Script 20:** Merges macro variables with stock-level data, ensuring proper temporal alignment

**Impact:** The addition of macroeconomic variables significantly improved model performance and provided economically meaningful insights. The model identified that macroeconomic conditions (inflation, default spreads) and market structure (liquidity) were more predictive than traditional momentum signals, particularly in the post-2016 regime.

---

## Current State

The project has evolved from a failed options prediction study to a robust stock return predictability framework that:
- Successfully merges CRSP and Compustat data using custom linking logic
- Implements rigorous data sanitization to ensure numerical stability
- Incorporates both stock-level and macroeconomic predictors
- Achieves economically significant out-of-sample predictive performance

