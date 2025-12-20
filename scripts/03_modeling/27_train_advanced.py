"""
Advanced Neural Network Training with State-of-the-Art Techniques

This script implements a comprehensive training pipeline with:
- Gradient clipping for stability
- Learning rate warmup and cosine annealing
- Multiple loss function options (MSE, Huber, IC)
- Gradient accumulation for larger effective batch sizes
- Model checkpointing with best weight restoration
- Comprehensive logging and metrics tracking
- Configurable training strategies via presets

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
from typing import Dict, List, Optional
import argparse

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from paths import get_processed_data_path, get_results_path
from training_config import TrainingConfig, get_config, list_presets
from training import (
    set_seed, clip_gradients, GradientClipper,
    get_scheduler, WarmupCosineScheduler,
    MixedPrecisionTrainer, EarlyStopping,
    ModelCheckpoint, TrainingLogger,
    train_epoch, evaluate
)
from losses import get_loss_function
from models import AssetPricingNet


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_advanced(config: TrainingConfig = None, verbose: bool = True):
    """
    Train neural network ensemble with advanced techniques.
    
    Args:
        config: Training configuration (uses default if None)
        verbose: Whether to print progress
    
    Returns:
        Dictionary with training results
    """
    if config is None:
        config = get_config('default')
    
    # Set random seed
    set_seed(config.random_seed)
    
    # Paths
    FILE_PATH = get_processed_data_path("thesis_dataset_macro.parquet")
    RESULTS_FOLDER = get_results_path("")
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("=" * 70)
        print("ADVANCED NEURAL NETWORK TRAINING")
        print("=" * 70)
        print(f"\n{config}")
        print()
    
    # ========================================================================
    # DATA LOADING AND PREPROCESSING
    # ========================================================================
    
    if verbose:
        print("Loading and preprocessing data...")
    
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
    
    # Time splits
    dates = df_liquid['date'].dt.year.values
    train_mask = dates <= config.train_end_year
    val_mask = (dates > config.train_end_year) & (dates <= config.val_end_year)
    test_mask = dates > config.val_end_year
    
    if verbose:
        print(f"\nData Split:")
        print(f"  Train: {train_mask.sum():,} obs (1996-{config.train_end_year})")
        print(f"  Val:   {val_mask.sum():,} obs ({config.train_end_year+1}-{config.val_end_year})")
        print(f"  Test:  {test_mask.sum():,} obs ({config.val_end_year+1}-2024)")
        print(f"  Features: {len(features)}")
    
    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_liquid.loc[train_mask, features].values)
    X_val = scaler.transform(df_liquid.loc[val_mask, features].values)
    X_test = scaler.transform(df_liquid.loc[test_mask, features].values)
    
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    y_train = df_liquid.loc[train_mask, target].values.reshape(-1, 1)
    y_val = df_liquid.loc[val_mask, target].values.reshape(-1, 1)
    y_test = df_liquid.loc[test_mask, target].values.reshape(-1, 1)
    
    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    
    # Device selection
    if config.device == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(config.device)
    
    if verbose:
        print(f"\nDevice: {device}")
    
    # ========================================================================
    # TRAINING SETUP
    # ========================================================================
    
    # Loss function
    criterion = get_loss_function(config.loss_function, **config.loss_kwargs)
    
    # Initialize logging
    logger = TrainingLogger(
        log_dir=str(RESULTS_FOLDER / "training_logs"),
        use_tensorboard=config.use_tensorboard
    )
    
    # Initialize checkpoint manager
    checkpoint = None
    if config.checkpoint_dir:
        checkpoint = ModelCheckpoint(
            checkpoint_dir=config.checkpoint_dir,
            save_best_only=config.save_best_only,
            mode='max'
        )
    
    # ========================================================================
    # ENSEMBLE TRAINING
    # ========================================================================
    
    ensemble_preds = np.zeros((len(y_test_t), config.ensemble_models))
    feature_importances = np.zeros(len(features))
    all_metrics = []
    
    if verbose:
        print(f"\nTraining ensemble of {config.ensemble_models} models...")
        print("-" * 70)
    
    for model_idx in range(config.ensemble_models):
        # Set seed for this model
        set_seed(config.random_seed + model_idx)
        
        if verbose:
            print(f"\n--- Model {model_idx + 1}/{config.ensemble_models} ---")
        
        # Initialize model with config
        model = AssetPricingNet(
            input_dim=len(features),
            hidden_dims=config.hidden_dims,
            dropout_rates=config.dropout_rates
        ).to(device)
        
        # Optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Data loader
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True
        )
        
        # Learning rate scheduler
        scheduler = get_scheduler(
            optimizer, config,
            steps_per_epoch=len(train_loader)
        )
        
        # Gradient clipper
        gradient_clipper = None
        if config.gradient_clip_norm:
            gradient_clipper = GradientClipper(
                max_norm=config.gradient_clip_norm
            )
        
        # Mixed precision trainer
        mixed_precision = None
        if config.use_mixed_precision:
            mixed_precision = MixedPrecisionTrainer(
                enabled=True,
                device=str(device)
            )
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            delta=config.early_stopping_delta,
            mode='max'
        )
        
        # Training loop
        model_metrics = []
        best_val_r2 = float('-inf')
        
        for epoch in range(config.epochs):
            # Train epoch
            train_loss = train_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                config=config,
                gradient_clipper=gradient_clipper,
                mixed_precision=mixed_precision
            )
            
            # Validate
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val_t.to(device)).cpu().numpy()
                val_r2 = r2_score(y_val_t.numpy(), val_preds)
            
            # Update scheduler
            if scheduler is not None:
                if config.scheduler_type == 'reduce_on_plateau':
                    scheduler.step(val_r2)
                else:
                    scheduler.step()
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log metrics
            epoch_metrics = {
                'model': model_idx + 1,
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_r2': val_r2,
                'learning_rate': current_lr
            }
            
            if gradient_clipper:
                stats = gradient_clipper.get_stats()
                epoch_metrics['mean_grad_norm'] = stats.get('mean_grad_norm', 0)
            
            model_metrics.append(epoch_metrics)
            logger.log(epoch + 1, epoch_metrics)
            
            # Print progress
            if verbose and (epoch + 1) % config.print_every_n_epochs == 0:
                print(f"   Epoch {epoch+1:3d}: Loss={train_loss:.6f}, "
                      f"Val R²={val_r2:.4%}, LR={current_lr:.2e}")
            
            # Early stopping check
            if early_stopping(val_r2, model, epoch + 1):
                if verbose:
                    print(f"   >>> Early stopping at epoch {epoch+1}! "
                          f"Best Val R²: {early_stopping.best_score:.4%}")
                break
            
            # Save checkpoint
            if checkpoint and (epoch + 1) % config.save_every_n_epochs == 0:
                checkpoint.save(
                    model, optimizer, epoch + 1, val_r2,
                    config=config.to_dict(),
                    filename=f'model_{model_idx+1}_epoch_{epoch+1}.pt'
                )
        
        # Restore best weights
        early_stopping.restore_best_weights(model)
        
        # Final predictions on test set
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test_t.to(device)).cpu().numpy()
            ensemble_preds[:, model_idx] = test_preds.flatten()
        
        # Feature importance (gradient-based)
        X_sample = X_train_t[:1000].to(device).requires_grad_(True)
        output = model(X_sample)
        output.sum().backward()
        grads = X_sample.grad.abs().mean(dim=0).cpu().numpy()
        feature_importances += grads
        
        all_metrics.extend(model_metrics)
    
    # ========================================================================
    # FINAL EVALUATION
    # ========================================================================
    
    # Ensemble predictions
    final_preds = np.mean(ensemble_preds, axis=1)
    final_r2 = r2_score(y_test_t.numpy(), final_preds)
    
    if verbose:
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"\n>>> FINAL TEST R²: {final_r2:.4f} ({final_r2*100:.2f}%)")
        print(f"\nThis is TRUE out-of-sample performance on {config.val_end_year+1}-2024 data")
        print(f"that was NEVER used for training or validation.")
    
    # Feature importance
    avg_importance = feature_importances / config.ensemble_models
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': avg_importance
    }).sort_values('Importance', ascending=False)
    
    if verbose:
        print("\nTop 10 Features:")
        print("-" * 40)
        for _, row in importance_df.head(10).iterrows():
            print(f"  {row['Feature']:20s}: {row['Importance']:.6f}")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    # Save importance
    importance_df.to_csv(str(RESULTS_FOLDER / "advanced_importance.csv"), index=False)
    
    # Save metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(str(RESULTS_FOLDER / "advanced_training_metrics.csv"), index=False)
    
    # Save config
    config.save(str(RESULTS_FOLDER / "training_config.json"))
    
    # Save summary
    summary = {
        'test_r2': final_r2,
        'config_preset': 'custom',
        'loss_function': config.loss_function,
        'scheduler_type': config.scheduler_type,
        'gradient_clip_norm': config.gradient_clip_norm,
        'ensemble_models': config.ensemble_models,
        'n_features': len(features),
        'train_period': f'1996-{config.train_end_year}',
        'val_period': f'{config.train_end_year+1}-{config.val_end_year}',
        'test_period': f'{config.val_end_year+1}-2024'
    }
    pd.DataFrame([summary]).to_csv(str(RESULTS_FOLDER / "advanced_summary.csv"), index=False)
    
    # Close logger
    logger.save("advanced_training_log.csv")
    logger.close()
    
    # Plot training curves
    plot_training_curves(metrics_df, RESULTS_FOLDER, config)
    
    if verbose:
        print(f"\n[SUCCESS] Results saved to {RESULTS_FOLDER}")
        print("=" * 70)
    
    return {
        'test_r2': final_r2,
        'predictions': final_preds,
        'importance': importance_df,
        'metrics': metrics_df,
        'config': config
    }


def plot_training_curves(metrics_df: pd.DataFrame, results_folder: Path, 
                         config: TrainingConfig):
    """Plot training curves for all models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Average metrics across models
    avg_metrics = metrics_df.groupby('epoch').agg({
        'train_loss': ['mean', 'std'],
        'val_r2': ['mean', 'std'],
        'learning_rate': 'mean'
    }).reset_index()
    
    # Flatten column names
    avg_metrics.columns = ['epoch', 'loss_mean', 'loss_std', 'r2_mean', 'r2_std', 'lr']
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    ax.plot(avg_metrics['epoch'], avg_metrics['loss_mean'], 'b-', linewidth=2)
    ax.fill_between(
        avg_metrics['epoch'],
        avg_metrics['loss_mean'] - avg_metrics['loss_std'],
        avg_metrics['loss_mean'] + avg_metrics['loss_std'],
        alpha=0.3
    )
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss (Mean ± Std)')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Validation R²
    ax = axes[0, 1]
    ax.plot(avg_metrics['epoch'], avg_metrics['r2_mean'] * 100, 'g-', linewidth=2)
    ax.fill_between(
        avg_metrics['epoch'],
        (avg_metrics['r2_mean'] - avg_metrics['r2_std']) * 100,
        (avg_metrics['r2_mean'] + avg_metrics['r2_std']) * 100,
        alpha=0.3, color='green'
    )
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation R² (%)')
    ax.set_title('Validation R² (Mean ± Std)')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate
    ax = axes[1, 0]
    ax.plot(avg_metrics['epoch'], avg_metrics['lr'], 'r-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title(f'Learning Rate Schedule ({config.scheduler_type})')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Per-model Validation R²
    ax = axes[1, 1]
    for model_num in metrics_df['model'].unique():
        model_data = metrics_df[metrics_df['model'] == model_num]
        ax.plot(model_data['epoch'], model_data['val_r2'] * 100, 
                alpha=0.7, label=f'Model {model_num}')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation R² (%)')
    ax.set_title('Per-Model Validation R²')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Advanced Training Results\n'
                 f'Loss: {config.loss_function} | Grad Clip: {config.gradient_clip_norm}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(results_folder / "advanced_training_curves.png"), dpi=150)
    plt.close()


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Advanced Neural Network Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration Presets:
  default     - Balanced configuration for most use cases
  fast        - Quick experiments with fewer epochs
  robust      - More regularization and longer training
  huber       - Uses Huber loss for robustness to outliers
  ic_optimized - Optimized for Information Coefficient
  large_batch - Large effective batch size via gradient accumulation

Example:
  python 27_train_advanced.py --preset robust --epochs 100
  python 27_train_advanced.py --preset huber --gradient-clip 0.5
        """
    )
    
    parser.add_argument('--preset', type=str, default='default',
                        choices=list(list_presets().keys()),
                        help='Configuration preset to use')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--learning-rate', type=float, help='Override learning rate')
    parser.add_argument('--loss', type=str, help='Loss function (mse, huber, ic)')
    parser.add_argument('--gradient-clip', type=float, help='Gradient clipping norm')
    parser.add_argument('--ensemble', type=int, help='Number of ensemble models')
    parser.add_argument('--scheduler', type=str, help='LR scheduler type')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')
    
    args = parser.parse_args()
    
    # Build config with overrides
    overrides = {}
    if args.epochs:
        overrides['epochs'] = args.epochs
    if args.batch_size:
        overrides['batch_size'] = args.batch_size
    if args.learning_rate:
        overrides['learning_rate'] = args.learning_rate
    if args.loss:
        overrides['loss_function'] = args.loss
    if args.gradient_clip:
        overrides['gradient_clip_norm'] = args.gradient_clip
    if args.ensemble:
        overrides['ensemble_models'] = args.ensemble
    if args.scheduler:
        overrides['scheduler_type'] = args.scheduler
    if args.seed:
        overrides['random_seed'] = args.seed
    
    config = get_config(args.preset, **overrides)
    
    # Run training
    results = train_advanced(config, verbose=not args.quiet)
    
    return results


if __name__ == "__main__":
    main()

