"""
SHAP (SHapley Additive exPlanations) Analysis for Neural Network

This script computes SHAP values for model interpretability:
- Global feature importance via SHAP
- Local explanations for individual predictions
- Comparison with gradient-based importance
- Publication-quality visualizations

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
import warnings

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from paths import get_processed_data_path, get_results_path
from models import AssetPricingNet, set_seed, get_device, sanitize_features

# Check for SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("Warning: SHAP not installed. Run: pip install shap")

# --- SETTINGS ---
RANDOM_SEED = 42
set_seed(RANDOM_SEED)

FILE_PATH = get_processed_data_path("thesis_dataset_macro.parquet")
RESULTS_FOLDER = get_results_path("")
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

# Time Splits
TRAIN_END = 2013
VAL_END = 2016

# SHAP Settings
N_BACKGROUND_SAMPLES = 100  # Background samples for SHAP
N_EXPLAIN_SAMPLES = 500     # Samples to explain
BATCH_SIZE = 4096
EPOCHS = 30


class ModelWrapper:
    """
    Wrapper to make PyTorch model compatible with SHAP.
    Converts numpy arrays to tensors and handles device placement.
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def __call__(self, X):
        """Forward pass for SHAP."""
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        
        X = X.to(self.device)
        
        with torch.no_grad():
            output = self.model(X)
        
        return output.cpu().numpy()


def train_model_for_shap(X_train, y_train, X_val, y_val, n_features, device):
    """Train a single model for SHAP analysis."""
    print("Training model for SHAP analysis...")
    
    model = AssetPricingNet(n_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)
    criterion = nn.MSELoss()
    
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    
    best_val_r2 = float('-inf')
    best_state = None
    patience = 5
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(torch.tensor(X_val, dtype=torch.float32).to(device))
            val_r2 = r2_score(y_val, val_preds.cpu().numpy())
        
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Val R² = {val_r2:.4%}")
    
    model.load_state_dict(best_state)
    print(f"  Best Val R²: {best_val_r2:.4%}")
    
    return model


def compute_shap_values(model, X_background, X_explain, feature_names, device):
    """
    Compute SHAP values using DeepExplainer or KernelExplainer.
    """
    print(f"\nComputing SHAP values...")
    print(f"  Background samples: {len(X_background)}")
    print(f"  Samples to explain: {len(X_explain)}")
    
    # Wrap model for SHAP
    wrapped_model = ModelWrapper(model, device)
    
    # Use KernelExplainer (model-agnostic, works with any model)
    # DeepExplainer would be faster but has compatibility issues with some architectures
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        explainer = shap.KernelExplainer(
            wrapped_model,
            X_background,
            link="identity"
        )
        
        # Compute SHAP values
        shap_values = explainer.shap_values(X_explain, nsamples=100)
    
    # Create DataFrame with SHAP values
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    
    return shap_values, explainer, shap_df


def compute_mean_abs_shap(shap_values, feature_names):
    """Compute mean absolute SHAP values for global importance."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP_Importance': mean_abs_shap
    }).sort_values('SHAP_Importance', ascending=False)
    
    return importance_df


def plot_shap_summary(shap_values, X_explain, feature_names, output_path):
    """Create SHAP summary plot (beeswarm)."""
    plt.figure(figsize=(12, 10))
    
    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_names,
        show=False,
        max_display=20
    )
    
    plt.title("SHAP Feature Importance", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_shap_bar(importance_df, output_path, top_n=20):
    """Create bar plot of mean absolute SHAP values."""
    plt.figure(figsize=(10, 8))
    
    top_features = importance_df.head(top_n)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))
    
    plt.barh(
        range(len(top_features)),
        top_features['SHAP_Importance'].values[::-1],
        color=colors[::-1]
    )
    plt.yticks(range(len(top_features)), top_features['Feature'].values[::-1])
    
    plt.xlabel('Mean |SHAP Value|', fontweight='bold')
    plt.ylabel('Feature', fontweight='bold')
    plt.title(f'Top {top_n} Features by SHAP Importance', fontweight='bold', pad=15)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_shap_waterfall(shap_values, X_sample, feature_names, expected_value, 
                        sample_idx, output_path):
    """Create waterfall plot for a single prediction."""
    plt.figure(figsize=(12, 8))
    
    # Create Explanation object for the sample
    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=expected_value,
        data=X_sample[sample_idx],
        feature_names=feature_names
    )
    
    shap.waterfall_plot(explanation, show=False, max_display=15)
    
    plt.title(f'SHAP Waterfall - Sample {sample_idx}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def compare_with_gradient_importance(shap_importance, gradient_importance_path, 
                                      output_path):
    """Compare SHAP importance with gradient-based importance."""
    # Load gradient importance if available
    try:
        grad_df = pd.read_csv(gradient_importance_path)
        
        # Merge
        comparison = shap_importance.merge(
            grad_df[['Feature', 'Importance']].rename(
                columns={'Importance': 'Gradient_Importance'}
            ),
            on='Feature',
            how='outer'
        ).fillna(0)
        
        # Normalize both to [0, 1]
        comparison['SHAP_Normalized'] = comparison['SHAP_Importance'] / comparison['SHAP_Importance'].max()
        comparison['Gradient_Normalized'] = comparison['Gradient_Importance'] / comparison['Gradient_Importance'].max()
        
        # Plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Top 15 by each method
        top_shap = comparison.nlargest(15, 'SHAP_Importance')
        top_grad = comparison.nlargest(15, 'Gradient_Importance')
        
        # SHAP ranking
        axes[0].barh(range(len(top_shap)), 
                     top_shap['SHAP_Normalized'].values[::-1],
                     color='steelblue')
        axes[0].set_yticks(range(len(top_shap)))
        axes[0].set_yticklabels(top_shap['Feature'].values[::-1])
        axes[0].set_xlabel('Normalized Importance')
        axes[0].set_title('Top 15 by SHAP', fontweight='bold')
        
        # Gradient ranking
        axes[1].barh(range(len(top_grad)), 
                     top_grad['Gradient_Normalized'].values[::-1],
                     color='coral')
        axes[1].set_yticks(range(len(top_grad)))
        axes[1].set_yticklabels(top_grad['Feature'].values[::-1])
        axes[1].set_xlabel('Normalized Importance')
        axes[1].set_title('Top 15 by Gradient', fontweight='bold')
        
        plt.suptitle('SHAP vs Gradient-Based Feature Importance', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"  Saved comparison: {output_path}")
        
        # Rank correlation
        from scipy.stats import spearmanr
        corr, pval = spearmanr(
            comparison['SHAP_Normalized'], 
            comparison['Gradient_Normalized']
        )
        print(f"\n  Rank Correlation (SHAP vs Gradient): {corr:.3f} (p={pval:.4f})")
        
        return comparison
        
    except FileNotFoundError:
        print("  Gradient importance file not found. Skipping comparison.")
        return None


def main():
    """Main SHAP analysis function."""
    if not HAS_SHAP:
        print("ERROR: SHAP library required. Install with: pip install shap")
        return
    
    print("=" * 60)
    print("SHAP FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_parquet(str(FILE_PATH))
    
    # Filter to liquid stocks
    df['size_rank'] = df.groupby('date')['mkt_cap'].rank(pct=True)
    df_liquid = df[df['size_rank'] > 0.3].copy()
    
    # Define features
    exclude = ['permno', 'date', 'ticker', 'ret', 'future_ret', 'prc', 'shrout', 
               'vol', 'bid', 'ask', 'siccd', 'size_rank', 'year', 'match_year', 'fyear']
    numeric_cols = df_liquid.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in exclude]
    target = 'future_ret'
    
    print(f"Features: {len(features)}")
    
    # Sanitize
    df_liquid = sanitize_features(df_liquid, features)
    df_liquid = df_liquid.dropna(subset=[target])
    
    # Time splits
    dates = df_liquid['date'].dt.year.values
    train_mask = dates <= TRAIN_END
    val_mask = (dates > TRAIN_END) & (dates <= VAL_END)
    test_mask = dates > VAL_END
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_liquid.loc[train_mask, features].values)
    X_val = scaler.transform(df_liquid.loc[val_mask, features].values)
    X_test = scaler.transform(df_liquid.loc[test_mask, features].values)
    
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    y_train = df_liquid.loc[train_mask, target].values
    y_val = df_liquid.loc[val_mask, target].values
    y_test = df_liquid.loc[test_mask, target].values
    
    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    device = get_device()
    print(f"Device: {device}")
    
    # Train model
    model = train_model_for_shap(X_train, y_train, X_val, y_val, len(features), device)
    
    # Select samples for SHAP
    np.random.seed(RANDOM_SEED)
    bg_indices = np.random.choice(len(X_train), min(N_BACKGROUND_SAMPLES, len(X_train)), replace=False)
    explain_indices = np.random.choice(len(X_test), min(N_EXPLAIN_SAMPLES, len(X_test)), replace=False)
    
    X_background = X_train[bg_indices]
    X_explain = X_test[explain_indices]
    
    # Compute SHAP values
    shap_values, explainer, shap_df = compute_shap_values(
        model, X_background, X_explain, features, device
    )
    
    # Compute mean absolute SHAP
    importance_df = compute_mean_abs_shap(shap_values, features)
    
    print("\n" + "=" * 60)
    print("TOP 20 FEATURES BY SHAP IMPORTANCE")
    print("=" * 60)
    print(importance_df.head(20).to_string(index=False))
    
    # Save results
    importance_df.to_csv(str(RESULTS_FOLDER / "shap_importance.csv"), index=False)
    shap_df.to_csv(str(RESULTS_FOLDER / "shap_values_sample.csv"), index=False)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Bar plot
    plot_shap_bar(importance_df, str(RESULTS_FOLDER / "shap_importance_bar.png"))
    
    # Summary plot (beeswarm)
    try:
        plot_shap_summary(
            shap_values, X_explain, features,
            str(RESULTS_FOLDER / "shap_summary.png")
        )
    except Exception as e:
        print(f"  Warning: Could not create summary plot: {e}")
    
    # Waterfall plots for extreme predictions
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(X_explain, dtype=torch.float32).to(device))
        predictions = predictions.cpu().numpy().flatten()
    
    # Get expected value (mean prediction on background)
    with torch.no_grad():
        bg_preds = model(torch.tensor(X_background, dtype=torch.float32).to(device))
        expected_value = bg_preds.cpu().numpy().mean()
    
    # Highest prediction
    high_idx = np.argmax(predictions)
    try:
        plot_shap_waterfall(
            shap_values, X_explain, features, expected_value,
            high_idx, str(RESULTS_FOLDER / "shap_waterfall_high.png")
        )
    except Exception as e:
        print(f"  Warning: Could not create high waterfall plot: {e}")
    
    # Lowest prediction
    low_idx = np.argmin(predictions)
    try:
        plot_shap_waterfall(
            shap_values, X_explain, features, expected_value,
            low_idx, str(RESULTS_FOLDER / "shap_waterfall_low.png")
        )
    except Exception as e:
        print(f"  Warning: Could not create low waterfall plot: {e}")
    
    # Compare with gradient importance
    print("\nComparing with gradient-based importance...")
    gradient_path = RESULTS_FOLDER / "optimized_importance.csv"
    compare_with_gradient_importance(
        importance_df, str(gradient_path),
        str(RESULTS_FOLDER / "shap_vs_gradient_comparison.png")
    )
    
    print("\n" + "=" * 60)
    print("[SUCCESS] SHAP analysis complete!")
    print(f"Results saved to: {RESULTS_FOLDER}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

