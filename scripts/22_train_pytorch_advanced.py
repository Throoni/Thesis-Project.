import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- SETTINGS ---
FILE_PATH = "processed_data/thesis_dataset_macro.parquet"
RESULTS_FOLDER = "results"
TRAIN_CUTOFF = 2016
BATCH_SIZE = 2048
EPOCHS = 30
LEARNING_RATE = 0.0005

if not os.path.exists(RESULTS_FOLDER): os.makedirs(RESULTS_FOLDER)

# Define the Neural Network
class AssetPricingNet(nn.Module):
    def __init__(self, input_dim):
        super(AssetPricingNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.model(x)

def train_pytorch_advanced():
    print("Loading Data...")
    df = pd.read_parquet(FILE_PATH)
    
    # 1. Filter Liquid Stocks (Top 30%)
    print("Filtering for Liquid Stocks...")
    df['size_rank'] = df.groupby('date')['mkt_cap'].rank(pct=True)
    df_liquid = df[df['size_rank'] > 0.3].copy()
    
    # 2. Features
    exclude = ['permno', 'date', 'ticker', 'ret', 'future_ret', 'prc', 'shrout', 'vol', 'bid', 'ask', 'siccd', 'size_rank']
    # Select only columns that are ALREADY numeric to start with
    numeric_cols = df_liquid.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in exclude]
    target = 'future_ret'
    
    print("-" * 30)
    print(f"Training with {len(features)} Features")
    print("-" * 30)

    # 3. SANITIZATION (The Unbreakable Method)
    print("Sanitizing Data...")
    
    # A. Force Convert to Numeric (Coerce errors to NaN)
    # This fixes the "TypeError: ufunc isinf" crash
    for col in features:
        df_liquid[col] = pd.to_numeric(df_liquid[col], errors='coerce')
    
    # B. Now safe to check for Infinity
    df_liquid.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # C. Fill NaNs
    for col in features:
        median_val = df_liquid[col].median()
        df_liquid[col] = df_liquid[col].fillna(0 if pd.isna(median_val) else median_val)
        
        # Clip outliers
        lower = df_liquid[col].quantile(0.01)
        upper = df_liquid[col].quantile(0.99)
        df_liquid[col] = df_liquid[col].clip(lower, upper)
    
    # Drop rows with missing target
    df_liquid = df_liquid.dropna(subset=[target])
    
    # 4. Scaling
    print("Scaling...")
    scaler = StandardScaler()
    # Use .values to avoid index alignment issues
    X_scaled = scaler.fit_transform(df_liquid[features].values)
    y = df_liquid[target].values.reshape(-1, 1)
    
    # Final Safety Patch for Scaler Errors
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)

    # 5. Split
    dates = df_liquid['date'].dt.year.values
    train_mask = dates <= TRAIN_CUTOFF
    test_mask = dates > TRAIN_CUTOFF
    
    # Convert to Tensors
    X_train_t = torch.tensor(X_scaled[train_mask], dtype=torch.float32)
    y_train_t = torch.tensor(y[train_mask], dtype=torch.float32)
    X_test_t = torch.tensor(X_scaled[test_mask], dtype=torch.float32)
    y_test_t = torch.tensor(y[test_mask], dtype=torch.float32)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using Device: {device}")
    
    model = AssetPricingNet(len(features)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # --- TRAINING ---
    print("Starting Training...")
    train_loss_history = []
    test_r2_history = []
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(loader)
        train_loss_history.append(avg_train_loss)
        
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test_t.to(device)).cpu().numpy()
            test_preds = np.nan_to_num(test_preds) # Safety
            r2 = r2_score(y_test_t.numpy(), test_preds)
            test_r2_history.append(r2)
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_train_loss:.6f} | Test R2: {r2:.4%}")

    # Plot Learning Curve
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Training Loss')
    plt.title('Training Convergence')
    plt.xlabel('Epochs')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(test_r2_history, color='orange', label='Test R2')
    plt.title('Out-of-Sample Performance')
    plt.xlabel('Epochs')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, "pytorch_learning_curve.png"))
    print(f"[SUCCESS] Learning Curve saved.")

    # Feature Importance (Gradient-based)
    print("Calculating Deep Feature Importance...")
    # Use random sample of test set
    indices = np.random.choice(len(X_test_t), size=2000, replace=False)
    X_sample = X_test_t[indices].to(device).requires_grad_(True)
    
    output = model(X_sample)
    output.sum().backward()
    grads = X_sample.grad.abs().mean(dim=0).cpu().numpy()
    
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': grads
    }).sort_values(by='Importance', ascending=False)
    
    print("\n--- TOP PREDICTORS (PyTorch Gradient) ---")
    print(importance.head(10))
    
    importance.to_csv(os.path.join(RESULTS_FOLDER, "pytorch_importance.csv"), index=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance.head(15), palette='viridis')
    plt.title('Deep Neural Net Importance (Gradient-Based)')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, "pytorch_feature_importance.png"))

if __name__ == "__main__":
    train_pytorch_advanced()