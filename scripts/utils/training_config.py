"""
Training Configuration Module

Centralized configuration management for neural network training.
Provides presets for different training strategies.

Author: Thesis Project
Date: December 2024
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional
import json
from pathlib import Path


@dataclass
class TrainingConfig:
    """
    Comprehensive training configuration.
    
    All training hyperparameters in one place for reproducibility.
    """
    # === Basic Training ===
    epochs: int = 50
    batch_size: int = 4096
    learning_rate: float = 0.0005
    weight_decay: float = 1e-3
    
    # === Time Splits ===
    train_end_year: int = 2013
    val_end_year: int = 2016
    
    # === Ensemble ===
    ensemble_models: int = 5
    
    # === Advanced Training ===
    gradient_clip_norm: Optional[float] = 1.0
    gradient_accumulation_steps: int = 1
    use_mixed_precision: bool = False
    
    # === Learning Rate Schedule ===
    scheduler_type: str = 'warmup_cosine'  # 'reduce_on_plateau', 'cosine', 'one_cycle', 'warmup_cosine', 'none'
    warmup_epochs: int = 5
    warmup_start_factor: float = 0.1
    min_lr: float = 1e-7
    
    # === Early Stopping ===
    early_stopping_patience: int = 6
    early_stopping_delta: float = 0.0
    
    # === Loss Function ===
    loss_function: str = 'mse'  # 'mse', 'huber', 'ic', 'asymmetric', 'combined'
    loss_kwargs: Dict = field(default_factory=dict)
    
    # === Regularization ===
    dropout_rates: tuple = (0.4, 0.3, 0.2)
    hidden_dims: tuple = (64, 32, 16)
    use_batch_norm: bool = True
    
    # === Checkpointing ===
    checkpoint_dir: Optional[str] = None
    save_best_only: bool = True
    save_every_n_epochs: int = 10
    
    # === Logging ===
    log_dir: Optional[str] = None
    use_tensorboard: bool = False
    print_every_n_epochs: int = 1
    
    # === Reproducibility ===
    random_seed: int = 42
    
    # === Device ===
    device: str = 'auto'  # 'auto', 'cuda', 'mps', 'cpu'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        d = asdict(self)
        # Convert tuples to lists for JSON serialization
        d['dropout_rates'] = list(d['dropout_rates'])
        d['hidden_dims'] = list(d['hidden_dims'])
        return d
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'TrainingConfig':
        """Create from dictionary."""
        # Convert lists back to tuples
        if 'dropout_rates' in d:
            d['dropout_rates'] = tuple(d['dropout_rates'])
        if 'hidden_dims' in d:
            d['hidden_dims'] = tuple(d['hidden_dims'])
        return cls(**d)
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))
    
    def __str__(self) -> str:
        """Pretty print configuration."""
        lines = ["TrainingConfig:"]
        for key, value in self.to_dict().items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

def get_default_config() -> TrainingConfig:
    """Default configuration - balanced for most use cases."""
    return TrainingConfig()


def get_fast_config() -> TrainingConfig:
    """Fast configuration for quick experiments."""
    return TrainingConfig(
        epochs=20,
        batch_size=8192,
        ensemble_models=3,
        early_stopping_patience=4,
        scheduler_type='reduce_on_plateau',
        warmup_epochs=2
    )


def get_robust_config() -> TrainingConfig:
    """Robust configuration with more regularization."""
    return TrainingConfig(
        epochs=100,
        batch_size=2048,
        learning_rate=0.0001,
        weight_decay=1e-2,
        ensemble_models=10,
        gradient_clip_norm=0.5,
        early_stopping_patience=10,
        dropout_rates=(0.5, 0.4, 0.3),
        scheduler_type='warmup_cosine',
        warmup_epochs=10
    )


def get_huber_loss_config() -> TrainingConfig:
    """Configuration using Huber loss for robustness to outliers."""
    return TrainingConfig(
        loss_function='huber',
        loss_kwargs={'delta': 0.5},
        epochs=50,
        gradient_clip_norm=1.0
    )


def get_ic_optimized_config() -> TrainingConfig:
    """Configuration optimized for Information Coefficient."""
    return TrainingConfig(
        loss_function='ic',
        epochs=75,
        batch_size=4096,
        learning_rate=0.001,
        scheduler_type='warmup_cosine',
        warmup_epochs=5,
        gradient_clip_norm=1.0
    )


def get_mixed_precision_config() -> TrainingConfig:
    """Configuration for mixed precision training (CUDA only)."""
    return TrainingConfig(
        use_mixed_precision=True,
        batch_size=8192,  # Can use larger batches with mixed precision
        gradient_accumulation_steps=1,
        device='cuda'
    )


def get_large_batch_config() -> TrainingConfig:
    """Configuration for large effective batch size via accumulation."""
    return TrainingConfig(
        batch_size=2048,
        gradient_accumulation_steps=4,  # Effective batch size = 8192
        learning_rate=0.001,  # Higher LR for larger batches
        warmup_epochs=10,
        scheduler_type='warmup_cosine'
    )


# ============================================================================
# CONFIGURATION PRESETS
# ============================================================================

PRESETS = {
    'default': get_default_config,
    'fast': get_fast_config,
    'robust': get_robust_config,
    'huber': get_huber_loss_config,
    'ic_optimized': get_ic_optimized_config,
    'mixed_precision': get_mixed_precision_config,
    'large_batch': get_large_batch_config,
}


def get_config(preset: str = 'default', **overrides) -> TrainingConfig:
    """
    Get configuration by preset name with optional overrides.
    
    Args:
        preset: Preset name ('default', 'fast', 'robust', etc.)
        **overrides: Keyword arguments to override preset values
    
    Returns:
        TrainingConfig with preset values and overrides applied
    
    Example:
        config = get_config('robust', epochs=50, learning_rate=0.0001)
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESETS.keys())}")
    
    config = PRESETS[preset]()
    
    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config parameter: {key}")
    
    return config


def list_presets() -> Dict[str, str]:
    """List available configuration presets with descriptions."""
    descriptions = {
        'default': 'Balanced configuration for most use cases',
        'fast': 'Quick experiments with fewer epochs and models',
        'robust': 'More regularization and longer training',
        'huber': 'Uses Huber loss for robustness to outliers',
        'ic_optimized': 'Optimized for Information Coefficient',
        'mixed_precision': 'Uses FP16 for faster training (CUDA only)',
        'large_batch': 'Large effective batch size via gradient accumulation',
    }
    return descriptions

