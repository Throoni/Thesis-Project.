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
import pandas as pd
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


def prepare_features(df, target='future_ret'):
    """
    Prepare features for modeling.
    
    Args:
        df: Raw DataFrame
        target: Target column name
    
    Returns:
        df_clean: Cleaned DataFrame
        features: List of feature names
    """
    # Exclude non-feature columns
    exclude = ['permno', 'date', 'ticker', 'ret', 'future_ret', 'prc', 'shrout', 
               'vol', 'bid', 'ask', 'siccd', 'size_rank', 'year', 'match_year', 'fyear']
    
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in exclude]
    
    # Sanitize
    df_clean = sanitize_features(df.copy(), features)
    df_clean = df_clean.dropna(subset=[target])
    
    return df_clean, features


def create_time_splits(df, train_end=2013, val_end=2016):
    """
    Create train/val/test masks based on year.
    
    Args:
        df: DataFrame with 'date' column
        train_end: Last year of training (inclusive)
        val_end: Last year of validation (inclusive)
    
    Returns:
        train_mask, val_mask, test_mask: Boolean arrays
    """
    years = df['date'].dt.year.values
    
    train_mask = years <= train_end
    val_mask = (years > train_end) & (years <= val_end)
    test_mask = years > val_end
    
    return train_mask, val_mask, test_mask


# --- TRAINING UTILITIES ---

def clip_gradients(model: nn.Module, max_norm: float = 1.0) -> float:
    """
    Clip gradients to prevent exploding gradients.
    
    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
    
    Returns:
        Total gradient norm before clipping
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


class WarmupCosineScheduler:
    """
    Learning rate scheduler with linear warmup followed by cosine annealing.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs
        total_epochs: Total training epochs
        min_lr: Minimum learning rate
    """
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, 
                 min_lr: float = 1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0
    
    def step(self):
        """Update learning rate for next epoch."""
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            factor = self.current_epoch / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            factor = 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = max(self.min_lr, base_lr * factor)
    
    def get_lr(self) -> list:
        """Get current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]


def compute_gradient_stats(model: nn.Module) -> dict:
    """
    Compute statistics about gradients.
    
    Args:
        model: PyTorch model with gradients computed
    
    Returns:
        Dictionary with gradient statistics
    """
    total_norm = 0.0
    param_count = 0
    max_grad = 0.0
    
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            param_count += 1
            max_grad = max(max_grad, p.grad.data.abs().max().item())
    
    total_norm = total_norm ** 0.5
    
    return {
        'total_norm': total_norm,
        'mean_norm': total_norm / max(param_count, 1),
        'max_grad': max_grad,
        'param_count': param_count
    }


def count_parameters(model: nn.Module) -> dict:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable
    }
