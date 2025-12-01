import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance

# --- SETTINGS ---
FILE_PATH = "processed_data/thesis_dataset_macro.parquet"
RESULTS_FOLDER = "results"
TRAIN_CUTOFF = 2016

if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

def train_final_macro():
    print("Loading Final Dataset with Macro...")
    df = pd.read_parquet(FILE_PATH)
    
    # 1. FILTER: Focus on Liquid Stocks (Top 30%)
    print("Filtering for Liquid Stocks...")
    df['size_rank'] = df.groupby('date')['mkt_cap'].rank(pct=True)
    df_liquid = df[df['size_rank'] > 0.3].copy()
    
    # 2. DEFINE FEATURES
    # Auto-select numeric columns, exclude IDs
    exclude = ['permno', 'date', 'ticker', 'ret', 'future_ret', 'prc', 'shrout', 'vol', 'bid', 'ask', 'siccd', 'size_rank']
    numeric_cols = df_liquid.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in exclude]
    target = 'future_ret'
    
    print("-" * 30)
    print(f"Training with {len(features)} Features (Stock + Macro):")
    print(features)
    print("-" * 30)
    
    # 3. SANITIZATION (The Safety Block)
    print("Sanitizing Data...")
    for col in features:
        # Force numeric
        df_liquid[col] = pd.to_numeric(df_liquid[col], errors='coerce')
        # Replace Inf
        df_liquid[col] = df_liquid[col].replace([np.inf, -np.inf], np.nan)
        # Fill NaN
        median_val = df_liquid[col].median()
        df_liquid[col] = df_liquid[col].fillna(0 if pd.isna(median_val) else median_val)
        # Winsorize
        lower = df_liquid[col].quantile(0.01)
        upper = df_liquid[col].quantile(0.99)
        df_liquid[col] = df_liquid[col].clip(lower, upper)

    df_liquid = df_liquid.dropna(subset=[target])

    # 4. SCALING
    print("Scaling...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_liquid[features].values)
    y = df_liquid[target].values
    
    # Final Safety Patch for Scaler Errors
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)

    # 5. SPLIT
    dates = df_liquid['date'].dt.year.values
    train_mask = dates <= TRAIN_CUTOFF
    test_mask = dates > TRAIN_CUTOFF
    
    X_train, y_train = X_scaled[train_mask], y[train_mask]
    X_test, y_test = X_scaled[test_mask], y[test_mask]
    
    print(f"Training on {len(X_train):,} rows...")
    
    # 6. TRAIN (Neural Network)
    print("Training Neural Network (Macro-Enabled)...")
    # Slightly larger network to handle the interactions
    nn = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16), # Deeper/Wider
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=50,
        random_state=42,
        early_stopping=True
    )
    nn.fit(X_train, y_train)
    
    # 7. EVALUATE
    pred_test = nn.predict(X_test)
    r2 = r2_score(y_test, pred_test)
    print(f"\n>>> Out-of-Sample R^2 (Macro Model): {r2:.4f} ({r2*100:.2f}%)")
    
    # 8. IMPORTANCE
    print("Calculating Importance...")
    limit = min(5000, len(X_test))
    r = permutation_importance(nn, X_test[:limit], y_test[:limit], n_repeats=5, random_state=42, n_jobs=-1)
    
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': r.importances_mean
    }).sort_values(by='Importance', ascending=False)
    
    print("\n--- TOP 10 PREDICTORS ---")
    print(importance.head(10))
    
    importance.to_csv(os.path.join(RESULTS_FOLDER, "macro_feature_importance.csv"), index=False)
    
    # 9. PLOT
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance.head(20), palette='magma')
    plt.title(f'Neural Net Importance (Stock + Macro)\nOOS R^2: {r2*100:.2f}%')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, "macro_importance.png"))
    print(f"[SUCCESS] Results saved to {RESULTS_FOLDER}")

if __name__ == "__main__":
    train_final_macro()