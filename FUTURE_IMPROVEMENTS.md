# Future Improvements Guide

This document provides detailed implementation guides for optional improvements to enhance the thesis project further.

**Priority Legend:**
- 🔴 HIGH: Significant improvement to methodology
- 🟡 MEDIUM: Nice to have for robustness
- 🟢 LOW: Polish and extras

---

## Table of Contents

1. [Permutation Importance](#1-permutation-importance) 🔴
2. [Consolidate Model Code (DRY)](#2-consolidate-model-code-dry) 🟡
3. [Unit Tests](#3-unit-tests) 🟡
4. [SHAP Values for Interpretability](#4-shap-values-for-interpretability) 🔴
5. [Rolling Window Training](#5-rolling-window-training) 🔴
6. [Cross-Validation](#6-cross-validation) 🟡
7. [Hyperparameter Tuning](#7-hyperparameter-tuning) 🟡
8. [Benchmark Models](#8-benchmark-models) 🔴
9. [Confidence Intervals](#9-confidence-intervals) 🟡
10. [Feature Selection](#10-feature-selection) 🟢

---

## 1. Permutation Importance

**Priority:** 🔴 HIGH  
**Why:** Gradient-based importance can be misleading. Permutation importance measures actual predictive contribution.

### Implementation

Create a new file `scripts/04_evaluation/29_permutation_importance.py`:

```python
"""
Permutation Feature Importance for Neural Network

More robust than gradient-based importance because it measures
the actual drop in performance when a feature is randomly shuffled.

Author: Thesis Project
Date: December 2024
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
from tqdm import tqdm

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from paths import get_processed_data_path, get_results_path

# --- SETTINGS ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

FILE_PATH = get_processed_data_path("thesis_dataset_macro.parquet")
RESULTS_FOLDER = get_results_path("")
N_PERMUTATIONS = 10  # Number of permutations per feature

# Time Splits
TRAIN_END = 2013
VAL_END = 2016


class AssetPricingNet(nn.Module):
    """Neural network architecture (must match training script)."""
    def __init__(self, input_dim):
        super(AssetPricingNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.model(x)


def permutation_importance(model, X, y, features, n_permutations=10, device='cpu'):
    """
    Calculate permutation importance for each feature.
    
    For each feature:
    1. Compute baseline R² with original data
    2. Shuffle the feature column
    3. Compute new R²
    4. Importance = baseline R² - shuffled R²
    
    Higher importance = bigger drop in performance when shuffled.
    """
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    
    # Baseline performance
    with torch.no_grad():
        baseline_preds = model(X_tensor).cpu().numpy()
    baseline_r2 = r2_score(y, baseline_preds)
    
    importance_scores = {}
    
    print(f"\nCalculating permutation importance for {len(features)} features...")
    print(f"Baseline R²: {baseline_r2:.4%}")
    print()
    
    for i, feature in enumerate(tqdm(features, desc="Features")):
        r2_drops = []
        
        for _ in range(n_permutations):
            # Copy and shuffle one feature
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])
            
            # Predict with shuffled feature
            X_perm_tensor = torch.tensor(X_permuted, dtype=torch.float32).to(device)
            with torch.no_grad():
                perm_preds = model(X_perm_tensor).cpu().numpy()
            
            perm_r2 = r2_score(y, perm_preds)
            r2_drops.append(baseline_r2 - perm_r2)
        
        importance_scores[feature] = {
            'mean': np.mean(r2_drops),
            'std': np.std(r2_drops),
            'min': np.min(r2_drops),
            'max': np.max(r2_drops)
        }
    
    return importance_scores, baseline_r2


def main():
    print("=" * 60)
    print("PERMUTATION FEATURE IMPORTANCE")
    print("=" * 60)
    
    # Load and prepare data (same as training script)
    print("\nLoading data...")
    df = pd.read_parquet(str(FILE_PATH))
    
    df['size_rank'] = df.groupby('date')['mkt_cap'].rank(pct=True)
    df_liquid = df[df['size_rank'] > 0.3].copy()
    
    exclude = ['permno', 'date', 'ticker', 'ret', 'future_ret', 'prc', 'shrout', 
               'vol', 'bid', 'ask', 'siccd', 'size_rank', 'year', 'match_year', 'fyear']
    numeric_cols = df_liquid.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in exclude]
    target = 'future_ret'
    
    # Sanitization
    for col in features:
        df_liquid[col] = pd.to_numeric(df_liquid[col], errors='coerce')
        df_liquid[col] = df_liquid[col].replace([np.inf, -np.inf], np.nan)
        median_val = df_liquid[col].median()
        df_liquid[col] = df_liquid[col].fillna(0 if pd.isna(median_val) else median_val)
        lower = df_liquid[col].quantile(0.01)
        upper = df_liquid[col].quantile(0.99)
        df_liquid[col] = df_liquid[col].clip(lower, upper)
    
    df_liquid = df_liquid.dropna(subset=[target])
    
    # Split
    dates = df_liquid['date'].dt.year.values
    train_mask = dates <= TRAIN_END
    val_mask = (dates > TRAIN_END) & (dates <= VAL_END)
    test_mask = dates > VAL_END
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_liquid.loc[train_mask, features].values)
    X_test = scaler.transform(df_liquid.loc[test_mask, features].values)
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    y_train = df_liquid.loc[train_mask, target].values.reshape(-1, 1)
    y_test = df_liquid.loc[test_mask, target].values.flatten()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Train a single model for importance calculation
    print("\nTraining model for importance calculation...")
    model = AssetPricingNet(len(features)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)
    criterion = nn.MSELoss()
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    
    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4096, shuffle=True)
    
    model.train()
    for epoch in range(30):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/30")
    
    # Calculate permutation importance on TEST set
    importance_scores, baseline_r2 = permutation_importance(
        model, X_test, y_test, features, 
        n_permutations=N_PERMUTATIONS, device=device
    )
    
    # Create DataFrame
    importance_df = pd.DataFrame([
        {
            'Feature': feat,
            'Importance': scores['mean'],
            'Std': scores['std'],
            'Min': scores['min'],
            'Max': scores['max']
        }
        for feat, scores in importance_scores.items()
    ]).sort_values('Importance', ascending=False)
    
    # Display top 20
    print("\n" + "=" * 60)
    print("TOP 20 FEATURES BY PERMUTATION IMPORTANCE")
    print("=" * 60)
    print(f"\nBaseline Test R²: {baseline_r2:.4%}")
    print()
    for i, row in importance_df.head(20).iterrows():
        print(f"{row['Feature']:25s}: {row['Importance']:>8.5f} ± {row['Std']:.5f}")
    
    # Save
    importance_df.to_csv(str(RESULTS_FOLDER / "permutation_importance.csv"), index=False)
    
    # Plot
    plt.figure(figsize=(10, 8))
    top_15 = importance_df.head(15)
    
    # Color by category
    macro_keywords = ['vix', 'inflation', 'cpi', 'term_spread', 'default_spread', 
                     'risk_free', 'sentiment']
    colors = ['#2E86AB' if any(kw in f.lower() for kw in macro_keywords) 
              else '#A23B72' for f in top_15['Feature']]
    
    bars = plt.barh(range(len(top_15)), top_15['Importance'], color=colors)
    plt.yticks(range(len(top_15)), top_15['Feature'])
    plt.xlabel('Importance (Drop in R² when permuted)')
    plt.title(f'Permutation Feature Importance\n(Baseline R² = {baseline_r2:.2%})')
    plt.gca().invert_yaxis()
    
    # Add error bars
    plt.errorbar(top_15['Importance'], range(len(top_15)), 
                xerr=top_15['Std'], fmt='none', color='black', capsize=3)
    
    # Legend
    from matplotlib.patches import Patch
    plt.legend(handles=[
        Patch(color='#2E86AB', label='Macro'),
        Patch(color='#A23B72', label='Stock-Level')
    ], loc='lower right')
    
    plt.tight_layout()
    plt.savefig(str(RESULTS_FOLDER / "permutation_importance.png"), dpi=150)
    plt.close()
    
    print(f"\n[SUCCESS] Saved to {RESULTS_FOLDER}")


if __name__ == "__main__":
    main()
```

### How to Run
```bash
cd "/Users/gord/Desktop/MBA 1SEM/Thesis_Project"
pip install tqdm  # If not installed
python scripts/04_evaluation/29_permutation_importance.py
```

### Interpretation
- Features with **higher importance** cause a bigger drop in R² when shuffled
- Error bars show stability across permutations
- More reliable than gradient-based importance

---

## 2. Consolidate Model Code (DRY)

**Priority:** 🟡 MEDIUM  
**Why:** The model architecture is duplicated in training and backtest scripts. Centralizing reduces bugs.

### Implementation

Create `scripts/utils/models.py`:

```python
"""
Shared model architectures and utilities.

This module provides common neural network architectures and training
utilities used across the project.

Author: Thesis Project
Date: December 2024
"""
import torch
import torch.nn as nn
import numpy as np
import copy


# --- REPRODUCIBILITY ---
def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- MODEL ARCHITECTURES ---
class AssetPricingNet(nn.Module):
    """
    Neural network for asset pricing / return prediction.
    
    Architecture:
        Input → Dense(64) + BN + SiLU + Dropout(0.4)
              → Dense(32) + BN + SiLU + Dropout(0.3)
              → Dense(16) + BN + SiLU + Dropout(0.2)
              → Dense(1) → Output
    
    Args:
        input_dim: Number of input features
        hidden_dims: Tuple of hidden layer dimensions (default: (64, 32, 16))
        dropout_rates: Tuple of dropout rates (default: (0.4, 0.3, 0.2))
    """
    def __init__(self, input_dim: int, hidden_dims=(64, 32, 16), 
                 dropout_rates=(0.4, 0.3, 0.2)):
        super(AssetPricingNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim, dropout in zip(hidden_dims, dropout_rates):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class EarlyStopping:
    """
    Early stopping handler with best weight restoration.
    
    Stops training if validation metric doesn't improve for `patience` epochs.
    Automatically saves and restores best model weights.
    
    Args:
        patience: Number of epochs to wait before stopping
        delta: Minimum improvement required
        mode: 'max' for metrics like R², 'min' for metrics like loss
    """
    def __init__(self, patience: int = 5, delta: float = 0, mode: str = 'max'):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None
    
    def __call__(self, score, model):
        if self.mode == 'max':
            is_improvement = self.best_score is None or score > self.best_score + self.delta
        else:
            is_improvement = self.best_score is None or score < self.best_score - self.delta
        
        if is_improvement:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    
    def restore_best(self, model):
        """Restore best weights to model."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# --- DATA UTILITIES ---
def sanitize_features(df, features):
    """
    Clean and sanitize feature columns.
    
    Steps:
        1. Convert to numeric
        2. Replace inf with NaN
        3. Fill NaN with median
        4. Winsorize at 1st/99th percentiles
    
    Args:
        df: DataFrame to modify (in-place)
        features: List of feature column names
    
    Returns:
        Modified DataFrame
    """
    import pandas as pd
    
    for col in features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        median_val = df[col].median()
        df[col] = df[col].fillna(0 if pd.isna(median_val) else median_val)
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(lower, upper)
    
    return df


def get_device():
    """Get best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
```

### How to Use

Update `23_train_optimized.py` imports:

```python
from models import AssetPricingNet, EarlyStopping, set_seed, sanitize_features, get_device

# Replace inline definitions with imports
set_seed(42)
device = get_device()
model = AssetPricingNet(len(features))
early_stopping = EarlyStopping(patience=6)
df_liquid = sanitize_features(df_liquid, features)
```

---

## 3. Unit Tests

**Priority:** 🟡 MEDIUM  
**Why:** Tests ensure critical functions work correctly after code changes.

### Implementation

Create `tests/test_models.py`:

```python
"""
Unit tests for model utilities and data processing.

Run with: pytest tests/ -v
"""
import pytest
import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "utils"))


class TestAssetPricingNet:
    """Tests for the neural network architecture."""
    
    def test_forward_pass_shape(self):
        """Test that output shape is correct."""
        from models import AssetPricingNet
        
        model = AssetPricingNet(input_dim=77)
        x = torch.randn(32, 77)  # Batch of 32, 77 features
        output = model(x)
        
        assert output.shape == (32, 1), f"Expected (32, 1), got {output.shape}"
    
    def test_forward_pass_no_nan(self):
        """Test that output contains no NaN values."""
        from models import AssetPricingNet
        
        model = AssetPricingNet(input_dim=50)
        x = torch.randn(100, 50)
        output = model(x)
        
        assert not torch.isnan(output).any(), "Output contains NaN values"
    
    def test_custom_architecture(self):
        """Test custom hidden dimensions."""
        from models import AssetPricingNet
        
        model = AssetPricingNet(
            input_dim=30,
            hidden_dims=(128, 64, 32),
            dropout_rates=(0.5, 0.4, 0.3)
        )
        x = torch.randn(16, 30)
        output = model(x)
        
        assert output.shape == (16, 1)


class TestEarlyStopping:
    """Tests for early stopping logic."""
    
    def test_early_stop_trigger(self):
        """Test that early stopping triggers after patience epochs."""
        from models import EarlyStopping
        
        es = EarlyStopping(patience=3, mode='max')
        model = torch.nn.Linear(10, 1)
        
        # Simulate decreasing scores
        scores = [0.5, 0.4, 0.3, 0.2, 0.1]
        for score in scores:
            es(score, model)
        
        assert es.early_stop, "Early stopping should have triggered"
    
    def test_no_early_stop_when_improving(self):
        """Test that early stopping doesn't trigger when improving."""
        from models import EarlyStopping
        
        es = EarlyStopping(patience=3, mode='max')
        model = torch.nn.Linear(10, 1)
        
        # Simulate improving scores
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        for score in scores:
            es(score, model)
        
        assert not es.early_stop, "Early stopping should not have triggered"
    
    def test_best_state_saved(self):
        """Test that best model state is saved."""
        from models import EarlyStopping
        
        es = EarlyStopping(patience=3, mode='max')
        model = torch.nn.Linear(10, 1)
        
        # Get initial weights
        initial_weight = model.weight.clone()
        
        es(0.5, model)  # Save state
        
        # Modify weights
        model.weight.data.fill_(999)
        
        # Restore
        es.restore_best(model)
        
        assert torch.allclose(model.weight, initial_weight), "Weights not restored correctly"


class TestDataSanitization:
    """Tests for data cleaning functions."""
    
    def test_sanitize_handles_inf(self):
        """Test that infinite values are handled."""
        from models import sanitize_features
        
        df = pd.DataFrame({
            'feature1': [1.0, np.inf, 3.0, -np.inf, 5.0],
            'feature2': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        result = sanitize_features(df.copy(), ['feature1', 'feature2'])
        
        assert not np.isinf(result['feature1']).any(), "Inf values not handled"
    
    def test_sanitize_handles_nan(self):
        """Test that NaN values are filled."""
        from models import sanitize_features
        
        df = pd.DataFrame({
            'feature1': [1.0, np.nan, 3.0, np.nan, 5.0]
        })
        
        result = sanitize_features(df.copy(), ['feature1'])
        
        assert not result['feature1'].isna().any(), "NaN values not filled"
    
    def test_winsorization(self):
        """Test that extreme values are clipped."""
        from models import sanitize_features
        
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000]  # 1000 is extreme
        })
        
        result = sanitize_features(df.copy(), ['feature1'])
        
        # 1000 should be clipped to 99th percentile
        assert result['feature1'].max() < 1000, "Extreme values not clipped"


class TestMomentumCalculation:
    """Tests for momentum feature calculation."""
    
    def test_momentum_skips_t1(self):
        """Test that 12-month momentum skips t-1 month."""
        # mom12m should be sum of returns from t-12 to t-2 (skip t-1)
        # Using shift(2).rolling(11).sum()
        
        returns = pd.Series([0.01] * 15)  # 15 months of 1% returns
        
        # Correct calculation: shift(2).rolling(11).sum()
        mom12m = returns.shift(2).rolling(11).sum()
        
        # At index 12, we should have sum of indices 1-11 (11 months)
        # Each is 0.01, so sum = 0.11
        expected = 0.11
        
        assert abs(mom12m.iloc[12] - expected) < 0.001, \
            f"Expected {expected}, got {mom12m.iloc[12]}"


class TestTrainTestSplit:
    """Tests for data splitting logic."""
    
    def test_no_data_leakage(self):
        """Test that train/val/test sets don't overlap."""
        dates = pd.Series([2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018])
        
        TRAIN_END = 2013
        VAL_END = 2016
        
        train_mask = dates <= TRAIN_END
        val_mask = (dates > TRAIN_END) & (dates <= VAL_END)
        test_mask = dates > VAL_END
        
        # No overlap
        assert not (train_mask & val_mask).any(), "Train and val overlap"
        assert not (train_mask & test_mask).any(), "Train and test overlap"
        assert not (val_mask & test_mask).any(), "Val and test overlap"
        
        # Complete coverage
        assert (train_mask | val_mask | test_mask).all(), "Some data not assigned"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### How to Run

```bash
cd "/Users/gord/Desktop/MBA 1SEM/Thesis_Project"
pip install pytest
pytest tests/ -v
```

---

## 4. SHAP Values for Interpretability

**Priority:** 🔴 HIGH  
**Why:** SHAP provides theoretically grounded, additive feature attributions that are more interpretable.

### Implementation

Create `scripts/04_evaluation/30_shap_analysis.py`:

```python
"""
SHAP (SHapley Additive exPlanations) Analysis

SHAP values provide:
- Theoretically grounded feature importance
- Direction of effect (positive/negative)
- Interaction effects between features

Author: Thesis Project
Date: December 2024
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "utils"))
from paths import get_processed_data_path, get_results_path

# Settings
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

FILE_PATH = get_processed_data_path("thesis_dataset_macro.parquet")
RESULTS_FOLDER = get_results_path("")
TRAIN_END = 2013
VAL_END = 2016


class AssetPricingNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.SiLU(), nn.Dropout(0.4),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(32, 16), nn.BatchNorm1d(16), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.model(x)


def main():
    print("=" * 60)
    print("SHAP ANALYSIS")
    print("=" * 60)
    
    # Load and prepare data
    print("\nLoading data...")
    df = pd.read_parquet(str(FILE_PATH))
    
    df['size_rank'] = df.groupby('date')['mkt_cap'].rank(pct=True)
    df_liquid = df[df['size_rank'] > 0.3].copy()
    
    exclude = ['permno', 'date', 'ticker', 'ret', 'future_ret', 'prc', 'shrout', 
               'vol', 'bid', 'ask', 'siccd', 'size_rank', 'year', 'match_year', 'fyear']
    numeric_cols = df_liquid.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in exclude]
    target = 'future_ret'
    
    # Sanitize
    for col in features:
        df_liquid[col] = pd.to_numeric(df_liquid[col], errors='coerce')
        df_liquid[col] = df_liquid[col].replace([np.inf, -np.inf], np.nan)
        median_val = df_liquid[col].median()
        df_liquid[col] = df_liquid[col].fillna(0 if pd.isna(median_val) else median_val)
        lower, upper = df_liquid[col].quantile(0.01), df_liquid[col].quantile(0.99)
        df_liquid[col] = df_liquid[col].clip(lower, upper)
    
    df_liquid = df_liquid.dropna(subset=[target])
    
    # Split
    from sklearn.preprocessing import StandardScaler
    dates = df_liquid['date'].dt.year.values
    train_mask = dates <= TRAIN_END
    test_mask = dates > VAL_END
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_liquid.loc[train_mask, features].values)
    X_test = scaler.transform(df_liquid.loc[test_mask, features].values)
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    y_train = df_liquid.loc[train_mask, target].values.reshape(-1, 1)
    
    # Train model
    print("Training model...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AssetPricingNet(len(features)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)
    criterion = nn.MSELoss()
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4096, shuffle=True)
    
    model.train()
    for epoch in range(30):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
    
    # SHAP Analysis
    print("\nCalculating SHAP values (this may take a few minutes)...")
    model.eval()
    model.cpu()  # SHAP works better on CPU
    
    # Use a subset of data for background (SHAP is computationally expensive)
    background_size = min(1000, len(X_train))
    background_indices = np.random.choice(len(X_train), background_size, replace=False)
    background = torch.tensor(X_train[background_indices], dtype=torch.float32)
    
    # Create explainer
    def model_predict(x):
        with torch.no_grad():
            return model(torch.tensor(x, dtype=torch.float32)).numpy()
    
    explainer = shap.KernelExplainer(model_predict, background.numpy())
    
    # Calculate SHAP values on test subset
    test_size = min(500, len(X_test))
    test_indices = np.random.choice(len(X_test), test_size, replace=False)
    X_test_subset = X_test[test_indices]
    
    shap_values = explainer.shap_values(X_test_subset, nsamples=100)
    
    # Summary plot
    print("\nGenerating SHAP plots...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_test_subset, feature_names=features, 
                     show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(str(RESULTS_FOLDER / "shap_summary.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Bar plot (mean absolute SHAP)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_subset, feature_names=features,
                     plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(str(RESULTS_FOLDER / "shap_importance.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save mean SHAP values
    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        'Feature': features,
        'Mean_SHAP': mean_shap
    }).sort_values('Mean_SHAP', ascending=False)
    shap_df.to_csv(str(RESULTS_FOLDER / "shap_importance.csv"), index=False)
    
    print("\nTop 15 Features by SHAP Importance:")
    print("-" * 40)
    for _, row in shap_df.head(15).iterrows():
        print(f"  {row['Feature']:25s}: {row['Mean_SHAP']:.6f}")
    
    print(f"\n[SUCCESS] SHAP results saved to {RESULTS_FOLDER}")


if __name__ == "__main__":
    main()
```

### How to Run
```bash
pip install shap
python scripts/04_evaluation/30_shap_analysis.py
```

---

## 5. Rolling Window Training

**Priority:** 🔴 HIGH  
**Why:** Tests model robustness across different time periods and mimics real trading.

### Implementation

Create `scripts/03_modeling/24_rolling_window_train.py`:

```python
"""
Rolling Window Training and Evaluation

Trains models on expanding or rolling windows to:
1. Mimic real-world trading conditions
2. Test stability across different market regimes
3. Produce more reliable out-of-sample estimates

Author: Thesis Project
Date: December 2024
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "utils"))
from paths import get_processed_data_path, get_results_path

# Settings
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

FILE_PATH = get_processed_data_path("thesis_dataset_macro.parquet")
RESULTS_FOLDER = get_results_path("")

# Rolling window settings
INITIAL_TRAIN_YEARS = 10  # Start with 10 years of training data
RETRAIN_FREQUENCY = 1     # Retrain every year
WINDOW_TYPE = 'expanding' # 'expanding' or 'rolling'
ROLLING_WINDOW_YEARS = 15 # Only used if WINDOW_TYPE == 'rolling'


class AssetPricingNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.SiLU(), nn.Dropout(0.4),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(32, 16), nn.BatchNorm1d(16), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(16, 1)
        )
    def forward(self, x): return self.model(x)


def train_model(X_train, y_train, n_features, device, epochs=30):
    """Train a single model and return it."""
    model = AssetPricingNet(n_features).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)
    criterion = nn.MSELoss()
    
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=4096, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
    
    return model


def main():
    print("=" * 60)
    print("ROLLING WINDOW TRAINING")
    print("=" * 60)
    print(f"\nWindow Type: {WINDOW_TYPE}")
    print(f"Initial Training Years: {INITIAL_TRAIN_YEARS}")
    print(f"Retrain Frequency: Every {RETRAIN_FREQUENCY} year(s)")
    
    # Load data
    print("\nLoading data...")
    df = pd.read_parquet(str(FILE_PATH))
    
    df['size_rank'] = df.groupby('date')['mkt_cap'].rank(pct=True)
    df_liquid = df[df['size_rank'] > 0.3].copy()
    
    exclude = ['permno', 'date', 'ticker', 'ret', 'future_ret', 'prc', 'shrout', 
               'vol', 'bid', 'ask', 'siccd', 'size_rank', 'year', 'match_year', 'fyear']
    numeric_cols = df_liquid.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in exclude]
    target = 'future_ret'
    
    # Sanitize
    for col in features:
        df_liquid[col] = pd.to_numeric(df_liquid[col], errors='coerce')
        df_liquid[col] = df_liquid[col].replace([np.inf, -np.inf], np.nan)
        median_val = df_liquid[col].median()
        df_liquid[col] = df_liquid[col].fillna(0 if pd.isna(median_val) else median_val)
        lower, upper = df_liquid[col].quantile(0.01), df_liquid[col].quantile(0.99)
        df_liquid[col] = df_liquid[col].clip(lower, upper)
    
    df_liquid = df_liquid.dropna(subset=[target])
    df_liquid['year'] = df_liquid['date'].dt.year
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Get unique years
    years = sorted(df_liquid['year'].unique())
    start_year = years[0]
    first_test_year = start_year + INITIAL_TRAIN_YEARS
    
    print(f"\nData years: {years[0]} - {years[-1]}")
    print(f"First test year: {first_test_year}")
    
    # Rolling window evaluation
    results = []
    all_predictions = []
    
    test_years = [y for y in years if y >= first_test_year]
    
    for i, test_year in enumerate(test_years):
        # Determine training window
        if WINDOW_TYPE == 'expanding':
            train_start = start_year
        else:  # rolling
            train_start = max(start_year, test_year - ROLLING_WINDOW_YEARS)
        
        train_end = test_year - 1
        
        print(f"\n--- Period {i+1}/{len(test_years)}: Train {train_start}-{train_end}, Test {test_year} ---")
        
        # Split data
        train_mask = (df_liquid['year'] >= train_start) & (df_liquid['year'] <= train_end)
        test_mask = df_liquid['year'] == test_year
        
        if train_mask.sum() < 1000 or test_mask.sum() < 100:
            print(f"  Skipping: insufficient data")
            continue
        
        # Scale (fit on training only)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(df_liquid.loc[train_mask, features].values)
        X_test = scaler.transform(df_liquid.loc[test_mask, features].values)
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)
        
        y_train = df_liquid.loc[train_mask, target].values.reshape(-1, 1)
        y_test = df_liquid.loc[test_mask, target].values.flatten()
        
        # Train model
        model = train_model(X_train, y_train, len(features), device)
        
        # Predict
        model.eval()
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = model(X_test_t).cpu().numpy().flatten()
        
        # Evaluate
        r2 = r2_score(y_test, preds)
        
        print(f"  Train size: {train_mask.sum():,}, Test size: {test_mask.sum():,}")
        print(f"  Test R²: {r2:.4%}")
        
        results.append({
            'test_year': test_year,
            'train_start': train_start,
            'train_end': train_end,
            'train_size': train_mask.sum(),
            'test_size': test_mask.sum(),
            'r2': r2
        })
        
        # Store predictions for backtest
        for idx, (pred, actual) in enumerate(zip(preds, y_test)):
            all_predictions.append({
                'year': test_year,
                'prediction': pred,
                'actual': actual
            })
    
    # Summary
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("ROLLING WINDOW RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nAverage R² across periods: {results_df['r2'].mean():.4%}")
    print(f"Std Dev of R²:             {results_df['r2'].std():.4%}")
    print(f"Min R²:                    {results_df['r2'].min():.4%}")
    print(f"Max R²:                    {results_df['r2'].max():.4%}")
    
    # Save results
    results_df.to_csv(str(RESULTS_FOLDER / "rolling_window_results.csv"), index=False)
    
    # Plot R² over time
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(results_df['test_year'], results_df['r2'] * 100, color='#2E86AB')
    plt.axhline(y=results_df['r2'].mean() * 100, color='red', linestyle='--', 
                label=f'Mean: {results_df["r2"].mean()*100:.2f}%')
    plt.xlabel('Test Year')
    plt.ylabel('R² (%)')
    plt.title('Out-of-Sample R² by Year')
    plt.legend()
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    plt.plot(results_df['test_year'], results_df['r2'].cumsum() / range(1, len(results_df)+1) * 100, 
             marker='o', color='#A23B72', linewidth=2)
    plt.xlabel('Test Year')
    plt.ylabel('Cumulative Average R² (%)')
    plt.title('Cumulative Average R² Over Time')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(str(RESULTS_FOLDER / "rolling_window_r2.png"), dpi=150)
    plt.close()
    
    print(f"\n[SUCCESS] Results saved to {RESULTS_FOLDER}")


if __name__ == "__main__":
    main()
```

### How to Run
```bash
python scripts/03_modeling/24_rolling_window_train.py
```

---

## 6. Cross-Validation

**Priority:** 🟡 MEDIUM  
**Why:** Time-series cross-validation provides more robust performance estimates.

### Implementation

Add to `scripts/03_modeling/23_train_optimized.py`:

```python
from sklearn.model_selection import TimeSeriesSplit

def time_series_cv(df, features, target, n_splits=5):
    """
    Perform time-series cross-validation.
    
    Uses sklearn's TimeSeriesSplit which ensures no future data
    leaks into training.
    """
    # Sort by date
    df = df.sort_values('date')
    
    X = df[features].values
    y = df[target].values
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\nFold {fold + 1}/{n_splits}")
        
        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        
        y_train = y[train_idx].reshape(-1, 1)
        y_val = y[val_idx]
        
        # Train model (simplified for CV)
        model = AssetPricingNet(len(features)).to(device)
        # ... train model ...
        
        # Evaluate
        r2 = r2_score(y_val, predictions)
        cv_scores.append(r2)
        print(f"  R²: {r2:.4%}")
    
    print(f"\nCV Mean R²: {np.mean(cv_scores):.4%} ± {np.std(cv_scores):.4%}")
    return cv_scores
```

---

## 7. Hyperparameter Tuning

**Priority:** 🟡 MEDIUM  
**Why:** Optimize model configuration for better performance.

### Implementation

Create `scripts/03_modeling/25_hyperparameter_tuning.py`:

```python
"""
Hyperparameter Tuning using Optuna

Optimizes:
- Learning rate
- Hidden layer dimensions
- Dropout rates
- Batch size
- Weight decay

Author: Thesis Project
Date: December 2024
"""
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "utils"))
from paths import get_processed_data_path, get_results_path

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

FILE_PATH = get_processed_data_path("thesis_dataset_macro.parquet")
RESULTS_FOLDER = get_results_path("")
TRAIN_END = 2013
VAL_END = 2016


def create_model(trial, input_dim):
    """Create model with trial-suggested hyperparameters."""
    n_layers = trial.suggest_int('n_layers', 2, 4)
    layers = []
    
    prev_dim = input_dim
    for i in range(n_layers):
        hidden_dim = trial.suggest_int(f'hidden_dim_{i}', 16, 128, step=16)
        dropout = trial.suggest_float(f'dropout_{i}', 0.1, 0.5)
        
        layers.extend([
            nn.Linear(prev_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        ])
        prev_dim = hidden_dim
    
    layers.append(nn.Linear(prev_dim, 1))
    return nn.Sequential(*layers)


def objective(trial, X_train, y_train, X_val, y_val, device):
    """Optuna objective function."""
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [1024, 2048, 4096, 8192])
    
    # Create model
    model = create_model(trial, X_train.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    # Data loaders
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_train_t, y_train_t), 
                       batch_size=batch_size, shuffle=True)
    
    # Train
    model.train()
    for epoch in range(20):  # Fewer epochs for tuning
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
    
    # Evaluate on validation set
    model.eval()
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(X_val_t).cpu().numpy().flatten()
    
    r2 = r2_score(y_val.flatten(), preds)
    return r2


def main():
    print("=" * 60)
    print("HYPERPARAMETER TUNING")
    print("=" * 60)
    
    # Load and prepare data (abbreviated)
    df = pd.read_parquet(str(FILE_PATH))
    df['size_rank'] = df.groupby('date')['mkt_cap'].rank(pct=True)
    df_liquid = df[df['size_rank'] > 0.3].copy()
    
    exclude = ['permno', 'date', 'ticker', 'ret', 'future_ret', 'prc', 'shrout', 
               'vol', 'bid', 'ask', 'siccd', 'size_rank', 'year', 'match_year', 'fyear']
    numeric_cols = df_liquid.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in exclude]
    target = 'future_ret'
    
    # Sanitize
    for col in features:
        df_liquid[col] = pd.to_numeric(df_liquid[col], errors='coerce')
        df_liquid[col] = df_liquid[col].replace([np.inf, -np.inf], np.nan)
        df_liquid[col] = df_liquid[col].fillna(df_liquid[col].median())
        df_liquid[col] = df_liquid[col].clip(
            df_liquid[col].quantile(0.01), df_liquid[col].quantile(0.99))
    df_liquid = df_liquid.dropna(subset=[target])
    
    dates = df_liquid['date'].dt.year.values
    train_mask = dates <= TRAIN_END
    val_mask = (dates > TRAIN_END) & (dates <= VAL_END)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_liquid.loc[train_mask, features].values)
    X_val = scaler.transform(df_liquid.loc[val_mask, features].values)
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)
    
    y_train = df_liquid.loc[train_mask, target].values.reshape(-1, 1)
    y_val = df_liquid.loc[val_mask, target].values.reshape(-1, 1)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Create study
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, device),
        n_trials=50,
        show_progress_bar=True
    )
    
    # Results
    print("\n" + "=" * 60)
    print("BEST HYPERPARAMETERS")
    print("=" * 60)
    print(f"\nBest Validation R²: {study.best_value:.4%}")
    print("\nBest Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results_df = study.trials_dataframe()
    results_df.to_csv(str(RESULTS_FOLDER / "hyperparameter_tuning.csv"), index=False)
    
    # Plot optimization history
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(str(RESULTS_FOLDER / "optuna_history.png"))
    
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(str(RESULTS_FOLDER / "optuna_importance.png"))
    
    print(f"\n[SUCCESS] Results saved to {RESULTS_FOLDER}")


if __name__ == "__main__":
    main()
```

### How to Run
```bash
pip install optuna plotly kaleido
python scripts/03_modeling/25_hyperparameter_tuning.py
```

---

## 8. Benchmark Models

**Priority:** 🔴 HIGH  
**Why:** Compare neural network to simpler baselines to demonstrate value-add.

### Implementation

Create `scripts/03_modeling/26_benchmark_models.py`:

```python
"""
Benchmark Models Comparison

Compares neural network against:
1. Historical Mean (naive baseline)
2. Linear Regression (OLS)
3. Ridge Regression (regularized linear)
4. Random Forest
5. Gradient Boosting (XGBoost/LightGBM)

Author: Thesis Project
Date: December 2024
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import xgboost/lightgbm
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

sys.path.append(str(Path(__file__).parent.parent / "utils"))
from paths import get_processed_data_path, get_results_path

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

FILE_PATH = get_processed_data_path("thesis_dataset_macro.parquet")
RESULTS_FOLDER = get_results_path("")
TRAIN_END = 2013
VAL_END = 2016


def train_neural_net(X_train, y_train, X_test, n_features, device):
    """Train neural network (simplified version)."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    class Net(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.SiLU(), nn.Dropout(0.4),
                nn.Linear(64, 32), nn.BatchNorm1d(32), nn.SiLU(), nn.Dropout(0.3),
                nn.Linear(32, 16), nn.BatchNorm1d(16), nn.SiLU(), nn.Dropout(0.2),
                nn.Linear(16, 1)
            )
        def forward(self, x): return self.model(x)
    
    model = Net(n_features).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)
    criterion = nn.MSELoss()
    
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=4096, shuffle=True)
    
    model.train()
    for _ in range(30):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32).to(device))
        return preds.cpu().numpy().flatten()


def main():
    print("=" * 60)
    print("BENCHMARK MODELS COMPARISON")
    print("=" * 60)
    
    # Load and prepare data
    print("\nLoading data...")
    df = pd.read_parquet(str(FILE_PATH))
    
    df['size_rank'] = df.groupby('date')['mkt_cap'].rank(pct=True)
    df_liquid = df[df['size_rank'] > 0.3].copy()
    
    exclude = ['permno', 'date', 'ticker', 'ret', 'future_ret', 'prc', 'shrout', 
               'vol', 'bid', 'ask', 'siccd', 'size_rank', 'year', 'match_year', 'fyear']
    numeric_cols = df_liquid.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in exclude]
    target = 'future_ret'
    
    # Sanitize
    for col in features:
        df_liquid[col] = pd.to_numeric(df_liquid[col], errors='coerce')
        df_liquid[col] = df_liquid[col].replace([np.inf, -np.inf], np.nan)
        df_liquid[col] = df_liquid[col].fillna(df_liquid[col].median())
        df_liquid[col] = df_liquid[col].clip(
            df_liquid[col].quantile(0.01), df_liquid[col].quantile(0.99))
    df_liquid = df_liquid.dropna(subset=[target])
    
    dates = df_liquid['date'].dt.year.values
    train_mask = dates <= VAL_END  # Use train+val for training
    test_mask = dates > VAL_END
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_liquid.loc[train_mask, features].values)
    X_test = scaler.transform(df_liquid.loc[test_mask, features].values)
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    y_train = df_liquid.loc[train_mask, target].values.flatten()
    y_test = df_liquid.loc[test_mask, target].values.flatten()
    
    print(f"Train size: {len(y_train):,}")
    print(f"Test size:  {len(y_test):,}")
    
    # Define models
    models = {
        'Historical Mean': None,  # Special case
        'OLS': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.001),
        'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5, 
                                                n_jobs=-1, random_state=RANDOM_SEED),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                                       random_state=RANDOM_SEED),
    }
    
    if HAS_XGB:
        models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, max_depth=3, 
                                             random_state=RANDOM_SEED, n_jobs=-1)
    
    if HAS_LGB:
        models['LightGBM'] = lgb.LGBMRegressor(n_estimators=100, max_depth=3,
                                               random_state=RANDOM_SEED, n_jobs=-1,
                                               verbosity=-1)
    
    # Evaluate models
    results = []
    
    print("\nTraining and evaluating models...")
    print("-" * 50)
    
    for name, model in models.items():
        print(f"  {name}...", end=" ")
        
        if name == 'Historical Mean':
            preds = np.full_like(y_test, y_train.mean())
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        
        r2 = r2_score(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        
        results.append({
            'Model': name,
            'R2': r2,
            'MSE': mse,
            'RMSE': np.sqrt(mse)
        })
        
        print(f"R² = {r2:.4%}")
    
    # Add Neural Network
    print("  Neural Network...", end=" ")
    import torch
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    nn_preds = train_neural_net(X_train, y_train.reshape(-1, 1), X_test, 
                                len(features), device)
    nn_r2 = r2_score(y_test, nn_preds)
    nn_mse = mean_squared_error(y_test, nn_preds)
    
    results.append({
        'Model': 'Neural Network',
        'R2': nn_r2,
        'MSE': nn_mse,
        'RMSE': np.sqrt(nn_mse)
    })
    print(f"R² = {nn_r2:.4%}")
    
    # Summary
    results_df = pd.DataFrame(results).sort_values('R2', ascending=False)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print()
    print(results_df.to_string(index=False))
    
    # Save
    results_df.to_csv(str(RESULTS_FOLDER / "benchmark_comparison.csv"), index=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    colors = ['#2E86AB' if 'Neural' in m else '#A23B72' for m in results_df['Model']]
    bars = plt.barh(range(len(results_df)), results_df['R2'] * 100, color=colors)
    plt.yticks(range(len(results_df)), results_df['Model'])
    plt.xlabel('R² (%)')
    plt.title('Model Comparison: Out-of-Sample R²')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(str(RESULTS_FOLDER / "benchmark_comparison.png"), dpi=150)
    plt.close()
    
    print(f"\n[SUCCESS] Results saved to {RESULTS_FOLDER}")


if __name__ == "__main__":
    main()
```

### How to Run
```bash
pip install xgboost lightgbm  # Optional, for XGBoost/LightGBM
python scripts/03_modeling/26_benchmark_models.py
```

---

## 9. Confidence Intervals

**Priority:** 🟡 MEDIUM  
**Why:** Quantify uncertainty in predictions and performance metrics.

### Implementation

Add bootstrap confidence intervals to evaluation:

```python
def bootstrap_r2(y_true, y_pred, n_bootstrap=1000, confidence=0.95):
    """
    Calculate bootstrap confidence interval for R².
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
    
    Returns:
        (mean_r2, lower_bound, upper_bound)
    """
    n = len(y_true)
    r2_scores = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n, n, replace=True)
        r2 = r2_score(y_true[indices], y_pred[indices])
        r2_scores.append(r2)
    
    r2_scores = np.array(r2_scores)
    
    alpha = 1 - confidence
    lower = np.percentile(r2_scores, alpha/2 * 100)
    upper = np.percentile(r2_scores, (1 - alpha/2) * 100)
    
    return np.mean(r2_scores), lower, upper


# Usage in evaluation:
mean_r2, lower, upper = bootstrap_r2(y_test, predictions, n_bootstrap=1000)
print(f"R² = {mean_r2:.4%} (95% CI: [{lower:.4%}, {upper:.4%}])")
```

---

## 10. Feature Selection

**Priority:** 🟢 LOW  
**Why:** Reduce dimensionality and potentially improve generalization.

### Implementation

```python
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

def select_features(X_train, y_train, feature_names, k=30, method='f_regression'):
    """
    Select top k features using statistical tests.
    
    Args:
        X_train: Training features
        y_train: Training target
        feature_names: List of feature names
        k: Number of features to select
        method: 'f_regression' or 'mutual_info'
    
    Returns:
        selected_features: List of selected feature names
        selector: Fitted selector object
    """
    if method == 'f_regression':
        selector = SelectKBest(f_regression, k=k)
    else:
        selector = SelectKBest(mutual_info_regression, k=k)
    
    selector.fit(X_train, y_train.flatten())
    
    # Get selected feature names
    selected_mask = selector.get_support()
    selected_features = [f for f, s in zip(feature_names, selected_mask) if s]
    
    # Print scores
    scores = selector.scores_
    feature_scores = sorted(zip(feature_names, scores), key=lambda x: -x[1])
    
    print(f"\nTop {k} Features by {method}:")
    for feat, score in feature_scores[:k]:
        print(f"  {feat:25s}: {score:.4f}")
    
    return selected_features, selector
```

---

## Summary: Implementation Priority

| Improvement | Priority | Time Est. | Impact |
|-------------|----------|-----------|--------|
| Permutation Importance | 🔴 HIGH | 1 hour | Better feature understanding |
| SHAP Values | 🔴 HIGH | 2 hours | Publication-quality interpretability |
| Rolling Window | 🔴 HIGH | 1 hour | Robustness validation |
| Benchmark Models | 🔴 HIGH | 1 hour | Justify neural network choice |
| Consolidate Code (DRY) | 🟡 MEDIUM | 30 min | Maintainability |
| Unit Tests | 🟡 MEDIUM | 1 hour | Code reliability |
| Cross-Validation | 🟡 MEDIUM | 1 hour | Better performance estimates |
| Hyperparameter Tuning | 🟡 MEDIUM | 2 hours | Performance optimization |
| Confidence Intervals | 🟡 MEDIUM | 30 min | Uncertainty quantification |
| Feature Selection | 🟢 LOW | 30 min | Dimensionality reduction |

---

## Quick Start: Run All Improvements

```bash
cd "/Users/gord/Desktop/MBA 1SEM/Thesis_Project"

# Install additional dependencies
pip install tqdm shap optuna plotly kaleido xgboost lightgbm pytest

# Run improvements in order
python scripts/04_evaluation/29_permutation_importance.py
python scripts/04_evaluation/30_shap_analysis.py
python scripts/03_modeling/24_rolling_window_train.py
python scripts/03_modeling/26_benchmark_models.py

# Optional: Hyperparameter tuning (takes longer)
python scripts/03_modeling/25_hyperparameter_tuning.py

# Run tests
pytest tests/ -v
```

---

*End of Future Improvements Guide*

