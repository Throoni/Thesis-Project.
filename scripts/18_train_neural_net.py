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
FILE_PATH = "processed_data/thesis_dataset_final.parquet"
RESULTS_FOLDER = "results"
TRAIN_CUTOFF = 2016

if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

def train_neural_net():
    print("Loading Data...")
    df = pd.read_parquet(FILE_PATH)
    
    # 1. FILTER: Focus on Liquid Stocks (Top 30% by Market Cap)
    print("Filtering for Liquid Stocks (Top 30%)...")
    df['size_rank'] = df.groupby('date')['mkt_cap'].rank(pct=True)
    df_liquid = df[df['size_rank'] > 0.3].copy()
    
    # 2. DEFINE FEATURES
    # Added 'prc' to exclude list (Raw price is bad predictor)
    exclude = ['permno', 'date', 'ticker', 'ret', 'future_ret', 'prc', 'shrout', 'vol', 'bid', 'ask', 'siccd', 'size_rank']
    # Select only numeric columns
    numeric_cols = df_liquid.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in exclude]
    target = 'future_ret'
    
    print("-" * 30)
    print(f"Features ({len(features)}): {features}")
    print("-" * 30)
    
    # 3. SANITIZATION (Phase 1: Pandas Level)
    print("Sanitizing Data (Pandas Phase)...")
    
    for col in features:
        # Force numeric
        df_liquid[col] = pd.to_numeric(df_liquid[col], errors='coerce')
        
        # Replace Inf with NaN
        df_liquid[col] = df_liquid[col].replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with Median. If Median is NaN (empty col), fill with 0.
        median_val = df_liquid[col].median()
        if pd.isna(median_val):
            df_liquid[col] = df_liquid[col].fillna(0)
        else:
            df_liquid[col] = df_liquid[col].fillna(median_val)
            
        # Winsorize (Clip extreme outliers)
        lower = df_liquid[col].quantile(0.01)
        upper = df_liquid[col].quantile(0.99)
        df_liquid[col] = df_liquid[col].clip(lower, upper)

    # Drop rows where Target is missing
    df_liquid = df_liquid.dropna(subset=[target])

    # 4. SCALING
    print("Scaling Features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_liquid[features].values)
    y = df_liquid[target].values
    
    # 5. SANITIZATION (Phase 2: Numpy Level - The "Unbreakable" Patch)
    # If Scaler created NaNs (due to 0 variance), this fixes them.
    print("Applying Final Safety Patch (NaN to 0)...")
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # 6. SPLIT
    dates = df_liquid['date'].dt.year.values
    train_mask = dates <= TRAIN_CUTOFF
    test_mask = dates > TRAIN_CUTOFF
    
    X_train, y_train = X_scaled[train_mask], y[train_mask]
    X_test, y_test = X_scaled[test_mask], y[test_mask]
    
    print(f"Training on {len(X_train):,} liquid stocks...")
    
    # 7. TRAIN (Neural Network)
    print("Training Neural Network (NN3)...")
    # Using SGD ('adam') with Early Stopping
    nn = MLPRegressor(
        hidden_layer_sizes=(32, 16, 8),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=50,
        random_state=42,
        early_stopping=True
    )
    nn.fit(X_train, y_train)
    
    # 8. EVALUATE
    pred_test = nn.predict(X_test)
    r2 = r2_score(y_test, pred_test)
    print(f"\n>>> Out-of-Sample R^2 (Liquid Stocks): {r2:.4f} ({r2*100:.2f}%)")
    
    # 9. FEATURE IMPORTANCE
    print("\nCalculating Importance...")
    limit = min(5000, len(X_test))
    r = permutation_importance(nn, X_test[:limit], y_test[:limit], n_repeats=5, random_state=42, n_jobs=-1)
    
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': r.importances_mean
    }).sort_values(by='Importance', ascending=False)
    
    print("\n--- TOP PREDICTORS (Neural Net) ---")
    print(importance.head(10))
    
    importance.to_csv(os.path.join(RESULTS_FOLDER, "nn_feature_importance.csv"), index=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance.head(15), palette='magma')
    plt.title(f'Neural Net Importance (Top 30% Stocks)\nOOS R^2: {r2*100:.2f}%')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, "nn_importance.png"))
    print(f"[SUCCESS] Saved plot to {RESULTS_FOLDER}/nn_importance.png")

if __name__ == "__main__":
    train_neural_net()