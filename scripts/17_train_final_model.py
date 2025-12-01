import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# --- SETTINGS ---
FILE_PATH = "processed_data/thesis_dataset_final.parquet"
RESULTS_FOLDER = "results"
TRAIN_CUTOFF = 2016

if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

def train_final():
    print("Loading Final Dataset...")
    df = pd.read_parquet(FILE_PATH)
    
    # 1. Define Features
    # Exclude ID columns and the Target
    exclude = ['permno', 'date', 'ticker', 'ret', 'future_ret', 'prc', 'shrout', 'vol', 'bid', 'ask', 'siccd']
    features = [c for c in df.columns if c not in exclude]
    target = 'future_ret'
    
    print(f"Training with {len(features)} predictors: {features}")
    
    # 2. SANITIZATION (The Fix)
    print("Sanitizing Data (Removing Infinities)...")
    
    # Replace Infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Fill NaN with the median of that column (Simple Imputation)
    # We use the global median for speed/stability here
    for col in features:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        
        # Optional: Clip extreme outliers (Winsorization) to prevent value errors
        # Cap at 1st and 99th percentiles
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(lower, upper)

    # Final check for clean data
    if df[features].isnull().any().any() or np.isinf(df[features]).any().any():
        print("[ERROR] Data still contains NaNs or Infs after cleaning!")
        return

    # 3. Train/Test Split
    train_data = df[df['date'].dt.year <= TRAIN_CUTOFF]
    test_data = df[df['date'].dt.year > TRAIN_CUTOFF]
    
    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]
    
    print(f"Training on {len(X_train):,} rows...")
    print(f"Testing on  {len(X_test):,} rows...")
    
    # 4. Train Random Forest
    print("Training Model (Random Forest)...")
    rf = RandomForestRegressor(
        n_estimators=100,      
        max_depth=10,          
        min_samples_leaf=50,   
        n_jobs=-1,             
        random_state=42
    )
    rf.fit(X_train, y_train)
    
    # 5. Evaluate
    pred_test = rf.predict(X_test)
    r2 = r2_score(y_test, pred_test)
    print(f"\n>>> Out-of-Sample R^2: {r2:.4f} ({r2*100:.2f}%)")
    
    # 6. Save Results
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\n--- TOP 10 PREDICTORS ---")
    print(importance.head(10))
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance.head(20), palette='viridis')
    plt.title(f'Feature Importance: Stock Returns (OOS R^2: {r2*100:.2f}%)')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, "final_feature_importance.png"))
    print(f"\n[SUCCESS] Plot saved to {RESULTS_FOLDER}/final_feature_importance.png")

if __name__ == "__main__":
    train_final()