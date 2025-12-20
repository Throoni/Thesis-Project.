"""
Advanced Training Utilities for PyTorch Neural Networks

This module provides state-of-the-art training utilities including:
- Gradient clipping
- Mixed precision training
- Learning rate warmup and advanced schedulers
- Gradient accumulation
- Model checkpointing
- Training metrics logging

Author: Thesis Project
Date: December 2024
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import pandas as pd
import copy
import os
import json
from pathlib import Path
from typing import Optional, Dict, List, Callable, Union
from dataclasses import dataclass, field, asdict
import warnings


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TrainingConfig:
    """
    Configuration for neural network training.
    
    Centralizes all training hyperparameters for reproducibility.
    """
    # Basic training
    epochs: int = 50
    batch_size: int = 4096
    learning_rate: float = 0.0005
    weight_decay: float = 1e-3
    
    # Advanced training
    gradient_clip_norm: Optional[float] = 1.0
    gradient_accumulation_steps: int = 1
    use_mixed_precision: bool = False
    
    # Learning rate schedule
    scheduler_type: str = 'reduce_on_plateau'  # 'cosine', 'one_cycle', 'warmup_cosine'
    warmup_epochs: int = 5
    warmup_start_factor: float = 0.1
    
    # Early stopping
    early_stopping_patience: int = 6
    early_stopping_delta: float = 0.0
    
    # Loss function
    loss_function: str = 'mse'  # 'mse', 'huber', 'ic', etc.
    loss_kwargs: Dict = field(default_factory=dict)
    
    # Checkpointing
    checkpoint_dir: Optional[str] = None
    save_best_only: bool = True
    
    # Reproducibility
    random_seed: int = 42
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'TrainingConfig':
        return cls(**d)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


# ============================================================================
# GRADIENT CLIPPING
# ============================================================================

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


class GradientClipper:
    """
    Gradient clipping utility with logging.
    """
    def __init__(self, max_norm: float = 1.0, log_frequency: int = 100):
        self.max_norm = max_norm
        self.log_frequency = log_frequency
        self.step_count = 0
        self.clipped_count = 0
        self.total_norm_history = []
    
    def clip(self, model: nn.Module) -> float:
        """Clip gradients and track statistics."""
        total_norm = clip_gradients(model, self.max_norm)
        
        self.step_count += 1
        self.total_norm_history.append(total_norm.item() if torch.is_tensor(total_norm) else total_norm)
        
        if total_norm > self.max_norm:
            self.clipped_count += 1
        
        return total_norm
    
    def get_stats(self) -> Dict:
        """Get gradient clipping statistics."""
        if not self.total_norm_history:
            return {}
        
        norms = np.array(self.total_norm_history)
        return {
            'mean_grad_norm': float(np.mean(norms)),
            'max_grad_norm': float(np.max(norms)),
            'clip_ratio': self.clipped_count / max(self.step_count, 1),
            'total_steps': self.step_count
        }


# ============================================================================
# LEARNING RATE SCHEDULERS
# ============================================================================

class WarmupScheduler(_LRScheduler):
    """
    Learning rate warmup scheduler.
    
    Linearly increases learning rate from `start_factor * lr` to `lr`
    over `warmup_epochs` epochs.
    """
    def __init__(self, optimizer, warmup_epochs: int, start_factor: float = 0.1, 
                 last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.start_factor = start_factor
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch >= self.warmup_epochs:
            return [group['lr'] for group in self.optimizer.param_groups]
        
        # Linear warmup
        factor = self.start_factor + (1 - self.start_factor) * (
            self.last_epoch / self.warmup_epochs
        )
        return [base_lr * factor for base_lr in self.base_lrs]


class WarmupCosineScheduler(_LRScheduler):
    """
    Warmup + Cosine Annealing scheduler.
    
    First warms up, then decays following cosine curve.
    """
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 min_lr: float = 1e-7, warmup_start_factor: float = 0.1,
                 last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.warmup_start_factor = warmup_start_factor
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            factor = self.warmup_start_factor + (1 - self.warmup_start_factor) * (
                self.last_epoch / self.warmup_epochs
            )
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            factor = 0.5 * (1 + np.cos(np.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * factor 
                for base_lr in self.base_lrs
            ]


def get_scheduler(optimizer, config: TrainingConfig, steps_per_epoch: int = None):
    """
    Factory function to create learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        config: Training configuration
        steps_per_epoch: Number of steps per epoch (for OneCycleLR)
    
    Returns:
        Learning rate scheduler
    """
    scheduler_type = config.scheduler_type.lower()
    
    if scheduler_type == 'reduce_on_plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=2, factor=0.5
        )
    
    elif scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=1e-7
        )
    
    elif scheduler_type == 'cosine_warm_restarts':
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-7
        )
    
    elif scheduler_type == 'one_cycle':
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch required for OneCycleLR")
        return optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=config.learning_rate * 10,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch
        )
    
    elif scheduler_type == 'warmup':
        return WarmupScheduler(
            optimizer, 
            warmup_epochs=config.warmup_epochs,
            start_factor=config.warmup_start_factor
        )
    
    elif scheduler_type == 'warmup_cosine':
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=config.warmup_epochs,
            total_epochs=config.epochs,
            warmup_start_factor=config.warmup_start_factor
        )
    
    elif scheduler_type == 'none':
        return None
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


# ============================================================================
# MIXED PRECISION TRAINING
# ============================================================================

class MixedPrecisionTrainer:
    """
    Mixed precision training wrapper.
    
    Uses automatic mixed precision for faster training on supported devices.
    Falls back gracefully on unsupported devices.
    """
    def __init__(self, enabled: bool = True, device: str = 'cuda'):
        self.enabled = enabled and self._check_support(device)
        self.device = device
        self.scaler = None
        
        if self.enabled:
            if 'cuda' in device:
                from torch.cuda.amp import GradScaler
                self.scaler = GradScaler()
            # Note: MPS doesn't fully support mixed precision yet
    
    def _check_support(self, device: str) -> bool:
        """Check if mixed precision is supported on device."""
        if 'cuda' in device:
            return torch.cuda.is_available()
        elif 'mps' in device:
            # MPS has limited AMP support
            return False
        return False
    
    def autocast_context(self):
        """Get autocast context manager."""
        if self.enabled and 'cuda' in self.device:
            from torch.cuda.amp import autocast
            return autocast()
        else:
            # Return a no-op context manager
            from contextlib import nullcontext
            return nullcontext()
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision."""
        if self.enabled and self.scaler is not None:
            return self.scaler.scale(loss)
        return loss
    
    def step(self, optimizer):
        """Optimizer step with mixed precision handling."""
        if self.enabled and self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def unscale_gradients(self, optimizer):
        """Unscale gradients before clipping."""
        if self.enabled and self.scaler is not None:
            self.scaler.unscale_(optimizer)


# ============================================================================
# EARLY STOPPING
# ============================================================================

class EarlyStopping:
    """
    Early stopping handler with best weight restoration.
    
    Stops training if validation metric doesn't improve for `patience` epochs.
    Automatically saves and restores best model weights.
    """
    def __init__(self, patience: int = 5, delta: float = 0.0, mode: str = 'max',
                 restore_best: bool = True):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.restore_best = restore_best
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None
        self.best_epoch = 0
    
    def __call__(self, score: float, model: nn.Module, epoch: int = 0) -> bool:
        """
        Check if training should stop.
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.mode == 'max':
            is_improvement = self.best_score is None or score > self.best_score + self.delta
        else:
            is_improvement = self.best_score is None or score < self.best_score - self.delta
        
        if is_improvement:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def restore_best_weights(self, model: nn.Module):
        """Restore best weights to model."""
        if self.restore_best and self.best_state is not None:
            model.load_state_dict(self.best_state)
    
    def get_best_score(self) -> float:
        return self.best_score if self.best_score is not None else float('-inf')


# ============================================================================
# MODEL CHECKPOINTING
# ============================================================================

class ModelCheckpoint:
    """
    Model checkpointing utility.
    
    Saves model weights, optimizer state, and training metrics.
    """
    def __init__(self, checkpoint_dir: str, save_best_only: bool = True,
                 mode: str = 'max', verbose: bool = True):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.mode = mode
        self.verbose = verbose
        self.best_score = None
    
    def save(self, model: nn.Module, optimizer: optim.Optimizer, 
             epoch: int, score: float, config: dict = None,
             filename: str = None):
        """
        Save checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            epoch: Current epoch
            score: Validation score
            config: Optional training config
            filename: Optional custom filename
        """
        # Check if we should save
        if self.save_best_only:
            is_best = self._is_best(score)
            if not is_best:
                return
        
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'score': score,
            'config': config
        }
        
        # Determine filename
        if filename is None:
            if self.save_best_only:
                filename = 'best_model.pt'
            else:
                filename = f'checkpoint_epoch_{epoch}.pt'
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        
        if self.verbose:
            print(f"  [Checkpoint] Saved to {path}")
    
    def _is_best(self, score: float) -> bool:
        """Check if score is best so far."""
        if self.best_score is None:
            self.best_score = score
            return True
        
        if self.mode == 'max':
            is_best = score > self.best_score
        else:
            is_best = score < self.best_score
        
        if is_best:
            self.best_score = score
        
        return is_best
    
    def load(self, model: nn.Module, optimizer: optim.Optimizer = None,
             filename: str = 'best_model.pt') -> dict:
        """
        Load checkpoint.
        
        Returns:
            Checkpoint dictionary
        """
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint


# ============================================================================
# TRAINING METRICS LOGGER
# ============================================================================

class TrainingLogger:
    """
    Comprehensive training metrics logger.
    
    Logs training metrics to CSV and optionally TensorBoard.
    """
    def __init__(self, log_dir: str = None, use_tensorboard: bool = False):
        self.log_dir = Path(log_dir) if log_dir else None
        self.use_tensorboard = use_tensorboard
        self.metrics_history = []
        self.tb_writer = None
        
        if log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / 'tensorboard'))
            except ImportError:
                warnings.warn("TensorBoard not available. Logging to CSV only.")
                self.use_tensorboard = False
    
    def log(self, epoch: int, metrics: Dict[str, float]):
        """
        Log metrics for an epoch.
        
        Args:
            epoch: Current epoch
            metrics: Dictionary of metric name -> value
        """
        # Add epoch to metrics
        metrics_with_epoch = {'epoch': epoch, **metrics}
        self.metrics_history.append(metrics_with_epoch)
        
        # Log to TensorBoard
        if self.tb_writer:
            for name, value in metrics.items():
                self.tb_writer.add_scalar(name, value, epoch)
    
    def save(self, filename: str = 'training_log.csv'):
        """Save metrics history to CSV."""
        if self.log_dir and self.metrics_history:
            df = pd.DataFrame(self.metrics_history)
            df.to_csv(self.log_dir / filename, index=False)
    
    def get_history(self) -> pd.DataFrame:
        """Get metrics history as DataFrame."""
        return pd.DataFrame(self.metrics_history)
    
    def close(self):
        """Close TensorBoard writer."""
        if self.tb_writer:
            self.tb_writer.close()


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(model: nn.Module, 
                train_loader: DataLoader,
                optimizer: optim.Optimizer,
                criterion: nn.Module,
                device: torch.device,
                config: TrainingConfig,
                gradient_clipper: GradientClipper = None,
                mixed_precision: MixedPrecisionTrainer = None) -> float:
    """
    Train for one epoch with advanced techniques.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        config: Training configuration
        gradient_clipper: Optional gradient clipper
        mixed_precision: Optional mixed precision trainer
    
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    accumulation_steps = config.gradient_accumulation_steps
    
    for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        # Mixed precision forward pass
        if mixed_precision:
            with mixed_precision.autocast_context():
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss = loss / accumulation_steps  # Scale for accumulation
        else:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss = loss / accumulation_steps
        
        # Backward pass
        if mixed_precision:
            mixed_precision.scale_loss(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            # Unscale gradients before clipping (for mixed precision)
            if mixed_precision:
                mixed_precision.unscale_gradients(optimizer)
            
            # Gradient clipping
            if gradient_clipper:
                gradient_clipper.clip(model)
            elif config.gradient_clip_norm:
                clip_gradients(model, config.gradient_clip_norm)
            
            # Optimizer step
            if mixed_precision:
                mixed_precision.step(optimizer)
            else:
                optimizer.step()
            
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(model: nn.Module,
             data_loader: DataLoader,
             criterion: nn.Module,
             device: torch.device,
             return_predictions: bool = False) -> Union[float, tuple]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model
        data_loader: Data loader
        criterion: Loss function
        device: Device
        return_predictions: Whether to return predictions
    
    Returns:
        Average loss (and predictions if return_predictions=True)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            num_batches += 1
            
            if return_predictions:
                all_predictions.append(outputs.cpu())
                all_targets.append(batch_y.cpu())
    
    avg_loss = total_loss / num_batches
    
    if return_predictions:
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        return avg_loss, predictions, targets
    
    return avg_loss


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    # Make CUDA deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

