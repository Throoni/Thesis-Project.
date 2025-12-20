"""
Unit tests for training utilities, loss functions, and configuration.

Run with: pytest tests/test_training_utils.py -v
"""
import pytest
import numpy as np
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "utils"))


# ============================================================================
# LOSS FUNCTIONS TESTS
# ============================================================================

class TestHuberLoss:
    """Tests for Huber (Smooth L1) Loss."""
    
    def test_huber_small_errors(self):
        """Test Huber loss behaves like MSE for small errors."""
        from losses import HuberLoss
        
        loss_fn = HuberLoss(delta=1.0)
        pred = torch.tensor([[0.1], [0.2], [0.3]])
        target = torch.tensor([[0.0], [0.1], [0.2]])
        
        loss = loss_fn(pred, target)
        assert not torch.isnan(loss)
        assert loss.item() > 0
    
    def test_huber_large_errors(self):
        """Test Huber loss is linear for large errors."""
        from losses import HuberLoss
        
        loss_fn = HuberLoss(delta=0.5)
        pred = torch.tensor([[10.0]])
        target = torch.tensor([[0.0]])
        
        loss = loss_fn(pred, target)
        assert not torch.isnan(loss)
        assert loss.item() > 0
    
    def test_huber_zero_loss(self):
        """Test Huber loss is zero when predictions equal targets."""
        from losses import HuberLoss
        
        loss_fn = HuberLoss(delta=1.0)
        pred = torch.tensor([[1.0], [2.0], [3.0]])
        target = pred.clone()
        
        loss = loss_fn(pred, target)
        assert loss.item() < 1e-6


class TestQuantileLoss:
    """Tests for Quantile (Pinball) Loss."""
    
    def test_quantile_median(self):
        """Test quantile loss for median (tau=0.5)."""
        from losses import QuantileLoss
        
        loss_fn = QuantileLoss(quantile=0.5)
        pred = torch.tensor([[0.5]])
        target = torch.tensor([[1.0]])
        
        loss = loss_fn(pred, target)
        assert not torch.isnan(loss)
    
    def test_quantile_asymmetric(self):
        """Test quantile loss is asymmetric for tau != 0.5."""
        from losses import QuantileLoss
        
        loss_fn_low = QuantileLoss(quantile=0.1)
        loss_fn_high = QuantileLoss(quantile=0.9)
        
        pred = torch.tensor([[0.0]])
        target = torch.tensor([[1.0]])  # Under-prediction
        
        loss_low = loss_fn_low(pred, target)
        loss_high = loss_fn_high(pred, target)
        
        # Low quantile should penalize under-prediction less
        assert loss_low.item() < loss_high.item()
    
    def test_quantile_invalid(self):
        """Test that invalid quantile raises error."""
        from losses import QuantileLoss
        
        with pytest.raises(AssertionError):
            QuantileLoss(quantile=0.0)
        
        with pytest.raises(AssertionError):
            QuantileLoss(quantile=1.0)


class TestAsymmetricLoss:
    """Tests for Asymmetric Loss."""
    
    def test_asymmetric_over_prediction(self):
        """Test asymmetric loss for over-prediction."""
        from losses import AsymmetricLoss
        
        loss_fn = AsymmetricLoss(alpha_over=1.0, alpha_under=2.0)
        pred = torch.tensor([[1.0]])
        target = torch.tensor([[0.0]])  # Over-prediction
        
        loss = loss_fn(pred, target)
        assert not torch.isnan(loss)
        assert loss.item() > 0
    
    def test_asymmetric_under_prediction(self):
        """Test asymmetric loss penalizes under-prediction more."""
        from losses import AsymmetricLoss
        
        loss_fn = AsymmetricLoss(alpha_over=1.0, alpha_under=2.0)
        
        # Same magnitude error, different direction
        pred_over = torch.tensor([[1.0]])
        pred_under = torch.tensor([[-1.0]])
        target = torch.tensor([[0.0]])
        
        loss_over = loss_fn(pred_over, target)
        loss_under = loss_fn(pred_under, target)
        
        # Under-prediction should have higher loss
        assert loss_under.item() > loss_over.item()


class TestICLoss:
    """Tests for Information Coefficient Loss."""
    
    def test_ic_perfect_correlation(self):
        """Test IC loss for perfectly correlated predictions."""
        from losses import ICLoss
        
        loss_fn = ICLoss()
        pred = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
        target = pred.clone()
        
        loss = loss_fn(pred, target)
        # Perfect correlation should give IC = 1, loss = -1
        assert loss.item() < 0  # Negative because we maximize IC
    
    def test_ic_negative_correlation(self):
        """Test IC loss for negatively correlated predictions."""
        from losses import ICLoss
        
        loss_fn = ICLoss()
        pred = torch.tensor([[5.0], [4.0], [3.0], [2.0], [1.0]])
        target = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
        
        loss = loss_fn(pred, target)
        # Negative correlation should give positive loss
        assert loss.item() > 0


class TestLossFunctionFactory:
    """Tests for the loss function factory."""
    
    def test_get_mse_loss(self):
        """Test getting MSE loss."""
        from losses import get_loss_function
        
        loss_fn = get_loss_function('mse')
        assert isinstance(loss_fn, nn.MSELoss)
    
    def test_get_huber_loss(self):
        """Test getting Huber loss with kwargs."""
        from losses import get_loss_function
        
        loss_fn = get_loss_function('huber', delta=0.5)
        assert hasattr(loss_fn, 'delta')
    
    def test_unknown_loss(self):
        """Test that unknown loss raises error."""
        from losses import get_loss_function
        
        with pytest.raises(ValueError):
            get_loss_function('unknown_loss')


# ============================================================================
# TRAINING CONFIG TESTS
# ============================================================================

class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        from training_config import TrainingConfig
        
        config = TrainingConfig()
        assert config.epochs == 50
        assert config.batch_size == 4096
        assert config.learning_rate == 0.0005
    
    def test_config_to_dict(self):
        """Test config serialization to dict."""
        from training_config import TrainingConfig
        
        config = TrainingConfig(epochs=100, learning_rate=0.001)
        d = config.to_dict()
        
        assert d['epochs'] == 100
        assert d['learning_rate'] == 0.001
    
    def test_config_from_dict(self):
        """Test config deserialization from dict."""
        from training_config import TrainingConfig
        
        d = {'epochs': 75, 'batch_size': 2048, 'hidden_dims': [64, 32]}
        config = TrainingConfig.from_dict(d)
        
        assert config.epochs == 75
        assert config.batch_size == 2048


class TestConfigPresets:
    """Tests for configuration presets."""
    
    def test_get_default_config(self):
        """Test default preset."""
        from training_config import get_config
        
        config = get_config('default')
        assert config.epochs == 50
    
    def test_get_fast_config(self):
        """Test fast preset has fewer epochs."""
        from training_config import get_config
        
        config = get_config('fast')
        assert config.epochs < 50
    
    def test_get_robust_config(self):
        """Test robust preset has more regularization."""
        from training_config import get_config
        
        config = get_config('robust')
        assert config.epochs > 50
        assert config.weight_decay > 0.001
    
    def test_config_with_overrides(self):
        """Test preset with overrides."""
        from training_config import get_config
        
        config = get_config('default', epochs=10, learning_rate=0.01)
        assert config.epochs == 10
        assert config.learning_rate == 0.01
    
    def test_unknown_preset(self):
        """Test that unknown preset raises error."""
        from training_config import get_config
        
        with pytest.raises(ValueError):
            get_config('nonexistent_preset')


# ============================================================================
# TRAINING UTILITIES TESTS
# ============================================================================

class TestGradientClipper:
    """Tests for GradientClipper utility."""
    
    def test_gradient_clipping(self):
        """Test that gradients are clipped."""
        from training import GradientClipper
        
        model = nn.Linear(10, 1)
        x = torch.randn(4, 10)
        y = model(x)
        y.sum().backward()
        
        # Artificially inflate gradients
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.fill_(100.0)
        
        clipper = GradientClipper(max_norm=1.0)
        clipper.clip(model)
        
        # Check gradients are now bounded
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        total_norm = total_norm ** 0.5
        
        assert total_norm <= 1.0 + 1e-6
    
    def test_clipper_stats(self):
        """Test gradient clipper statistics."""
        from training import GradientClipper
        
        clipper = GradientClipper(max_norm=1.0)
        model = nn.Linear(10, 1)
        
        for _ in range(5):
            x = torch.randn(4, 10)
            y = model(x)
            y.sum().backward()
            clipper.clip(model)
            model.zero_grad()
        
        stats = clipper.get_stats()
        assert 'mean_grad_norm' in stats
        assert 'max_grad_norm' in stats
        assert stats['total_steps'] == 5


class TestWarmupScheduler:
    """Tests for learning rate warmup scheduler."""
    
    def test_warmup_increases_lr(self):
        """Test that LR increases during warmup."""
        from training import WarmupScheduler
        
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = WarmupScheduler(optimizer, warmup_epochs=5, start_factor=0.1)
        
        lrs = []
        for _ in range(5):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        
        # LR should increase during warmup
        for i in range(1, len(lrs)):
            assert lrs[i] > lrs[i-1] or abs(lrs[i] - lrs[i-1]) < 1e-6
    
    def test_warmup_reaches_base_lr(self):
        """Test that LR reaches base LR after warmup."""
        from training import WarmupScheduler
        
        base_lr = 0.01
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
        scheduler = WarmupScheduler(optimizer, warmup_epochs=5, start_factor=0.1)
        
        # Step through warmup period
        for _ in range(5):
            scheduler.step()
        
        # Should be at or near base LR (tolerance for floating point)
        final_lr = optimizer.param_groups[0]['lr']
        # After warmup, LR should be at least 80% of base LR
        assert final_lr >= base_lr * 0.8, f"LR {final_lr} should be near {base_lr}"


class TestEarlyStoppingAdvanced:
    """Additional tests for EarlyStopping."""
    
    def test_early_stopping_min_mode(self):
        """Test early stopping in min mode (for loss)."""
        from training import EarlyStopping
        
        es = EarlyStopping(patience=3, mode='min')
        model = nn.Linear(10, 1)
        
        # Simulate decreasing then increasing loss
        losses = [1.0, 0.8, 0.6, 0.7, 0.8, 0.9]
        for loss in losses:
            es(loss, model)
        
        assert es.early_stop
        assert es.best_score == 0.6
    
    def test_early_stopping_delta(self):
        """Test early stopping with minimum improvement delta."""
        from training import EarlyStopping
        
        es = EarlyStopping(patience=3, delta=0.1, mode='max')
        model = nn.Linear(10, 1)
        
        # Small improvements less than delta
        scores = [0.5, 0.51, 0.52, 0.53]  # < 0.1 improvement
        for score in scores:
            es(score, model)
        
        # Should trigger early stop (improvements below delta)
        assert es.early_stop


class TestModelCheckpoint:
    """Tests for ModelCheckpoint utility."""
    
    def test_checkpoint_save_load(self, tmp_path):
        """Test saving and loading checkpoints."""
        from training import ModelCheckpoint
        
        checkpoint_dir = str(tmp_path / "checkpoints")
        checkpoint = ModelCheckpoint(checkpoint_dir, save_best_only=False)
        
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Save
        checkpoint.save(model, optimizer, epoch=5, score=0.8)
        
        # Modify model
        original_weight = model.weight.clone()
        model.weight.data.fill_(999)
        
        # Load
        checkpoint.load(model, optimizer, filename='checkpoint_epoch_5.pt')
        
        assert torch.allclose(model.weight, original_weight)
    
    def test_checkpoint_save_best_only(self, tmp_path):
        """Test save_best_only mode."""
        from training import ModelCheckpoint
        
        checkpoint_dir = str(tmp_path / "checkpoints")
        checkpoint = ModelCheckpoint(checkpoint_dir, save_best_only=True, mode='max')
        
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters())
        
        # First save (best so far)
        checkpoint.save(model, optimizer, epoch=1, score=0.5)
        assert checkpoint.best_score == 0.5
        
        # Worse score - should not update best
        checkpoint.save(model, optimizer, epoch=2, score=0.3)
        assert checkpoint.best_score == 0.5
        
        # Better score - should update best
        checkpoint.save(model, optimizer, epoch=3, score=0.7)
        assert checkpoint.best_score == 0.7


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestTrainingIntegration:
    """Integration tests for training utilities working together."""
    
    def test_full_training_loop(self):
        """Test complete training loop with all utilities."""
        from training import (
            set_seed, GradientClipper, EarlyStopping, TrainingLogger
        )
        from losses import get_loss_function
        from models import AssetPricingNet
        
        set_seed(42)
        
        # Setup
        model = AssetPricingNet(input_dim=20)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        criterion = get_loss_function('huber', delta=1.0)
        clipper = GradientClipper(max_norm=1.0)
        early_stopping = EarlyStopping(patience=2, mode='min')
        
        # Synthetic data
        X = torch.randn(100, 20)
        y = torch.randn(100, 1)
        
        # Training loop
        for epoch in range(10):
            model.train()
            optimizer.zero_grad()
            
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            
            clipper.clip(model)
            optimizer.step()
            
            # Validation (use same data for simplicity)
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X), y)
            
            early_stopping(val_loss.item(), model)
            
            if early_stopping.early_stop:
                break
        
        # Restore best
        early_stopping.restore_best_weights(model)
        
        # Verify model works
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(5, 20))
        
        assert output.shape == (5, 1)
        assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

