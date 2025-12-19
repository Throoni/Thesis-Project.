"""
Unit tests for model utilities and data processing.

Run with: pytest tests/ -v
"""
import pytest
import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "utils"))

from models import AssetPricingNet, EarlyStopping, sanitize_features, get_device, set_seed


class TestAssetPricingNet:
    """Tests for the neural network architecture."""
    
    def test_forward_pass_shape(self):
        """Test that output shape is correct."""
        model = AssetPricingNet(input_dim=77)
        x = torch.randn(32, 77)  # Batch of 32, 77 features
        output = model(x)
        
        assert output.shape == (32, 1), f"Expected (32, 1), got {output.shape}"
    
    def test_forward_pass_no_nan(self):
        """Test that output contains no NaN values."""
        model = AssetPricingNet(input_dim=50)
        x = torch.randn(100, 50)
        output = model(x)
        
        assert not torch.isnan(output).any(), "Output contains NaN values"
    
    def test_custom_architecture(self):
        """Test custom hidden dimensions."""
        model = AssetPricingNet(
            input_dim=30,
            hidden_dims=(128, 64, 32),
            dropout_rates=(0.5, 0.4, 0.3)
        )
        x = torch.randn(16, 30)
        output = model(x)
        
        assert output.shape == (16, 1)
    
    def test_single_sample(self):
        """Test with batch size 1 (common edge case)."""
        model = AssetPricingNet(input_dim=10)
        model.eval()  # BatchNorm requires eval mode for single sample
        x = torch.randn(1, 10)
        output = model(x)
        
        assert output.shape == (1, 1)


class TestEarlyStopping:
    """Tests for early stopping logic."""
    
    def test_early_stop_trigger(self):
        """Test that early stopping triggers after patience epochs."""
        es = EarlyStopping(patience=3, mode='max')
        model = torch.nn.Linear(10, 1)
        
        # Simulate decreasing scores
        scores = [0.5, 0.4, 0.3, 0.2, 0.1]
        for score in scores:
            es(score, model)
        
        assert es.early_stop, "Early stopping should have triggered"
    
    def test_no_early_stop_when_improving(self):
        """Test that early stopping doesn't trigger when improving."""
        es = EarlyStopping(patience=3, mode='max')
        model = torch.nn.Linear(10, 1)
        
        # Simulate improving scores
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        for score in scores:
            es(score, model)
        
        assert not es.early_stop, "Early stopping should not have triggered"
    
    def test_best_state_saved(self):
        """Test that best model state is saved."""
        es = EarlyStopping(patience=3, mode='max')
        model = torch.nn.Linear(10, 1)
        
        # Get initial weights
        initial_weight = model.weight.clone()
        
        es(0.5, model)  # Save state
        
        # Modify weights
        model.weight.data.fill_(999)
        
        # Restore
        es.restore_best(model)
        
        assert torch.allclose(model.weight, initial_weight), "Weights not restored correctly"
    
    def test_min_mode(self):
        """Test early stopping in 'min' mode (for loss)."""
        es = EarlyStopping(patience=3, mode='min')
        model = torch.nn.Linear(10, 1)
        
        # Simulate decreasing loss (improvement)
        losses = [1.0, 0.8, 0.6, 0.4, 0.2]
        for loss in losses:
            es(loss, model)
        
        assert not es.early_stop, "Should not stop when loss is decreasing"
    
    def test_patience_reset_on_improvement(self):
        """Test that patience counter resets on improvement."""
        es = EarlyStopping(patience=3, mode='max')
        model = torch.nn.Linear(10, 1)
        
        # Score goes down for 2 epochs, then improves
        scores = [0.5, 0.4, 0.3, 0.6, 0.5, 0.4, 0.3]
        for score in scores:
            es(score, model)
        
        # Should trigger after 0.6 -> 0.5 -> 0.4 -> 0.3 (3 epochs of no improvement)
        assert es.early_stop


class TestDataSanitization:
    """Tests for data cleaning functions."""
    
    def test_sanitize_handles_inf(self):
        """Test that infinite values are handled."""
        df = pd.DataFrame({
            'feature1': [1.0, np.inf, 3.0, -np.inf, 5.0],
            'feature2': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        result = sanitize_features(df.copy(), ['feature1', 'feature2'])
        
        assert not np.isinf(result['feature1']).any(), "Inf values not handled"
    
    def test_sanitize_handles_nan(self):
        """Test that NaN values are filled."""
        df = pd.DataFrame({
            'feature1': [1.0, np.nan, 3.0, np.nan, 5.0]
        })
        
        result = sanitize_features(df.copy(), ['feature1'])
        
        assert not result['feature1'].isna().any(), "NaN values not filled"
    
    def test_winsorization(self):
        """Test that extreme values are clipped."""
        df = pd.DataFrame({
            'feature1': list(range(1, 101)) + [10000]  # 100 normal + 1 extreme
        })
        
        result = sanitize_features(df.copy(), ['feature1'])
        
        # 10000 should be clipped to 99th percentile
        assert result['feature1'].max() < 10000, "Extreme values not clipped"
    
    def test_preserves_normal_values(self):
        """Test that normal values are preserved."""
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        result = sanitize_features(df.copy(), ['feature1'])
        
        # Middle values should be unchanged
        assert result['feature1'].iloc[2] == 3.0


class TestMomentumCalculation:
    """Tests for momentum feature calculation."""
    
    def test_momentum_skips_t1(self):
        """Test that 12-month momentum skips t-1 month."""
        # mom12m should be sum of returns from t-12 to t-2 (skip t-1)
        # Using shift(2).rolling(11).sum()
        
        returns = pd.Series([0.01] * 15)  # 15 months of 1% returns
        
        # Correct calculation: shift(2).rolling(11).sum()
        mom12m = returns.shift(2).rolling(11).sum()
        
        # At index 12, we should have sum of indices 1-11 (11 months)
        # Each is 0.01, so sum = 0.11
        expected = 0.11
        
        assert abs(mom12m.iloc[12] - expected) < 0.001, \
            f"Expected {expected}, got {mom12m.iloc[12]}"
    
    def test_mom6m_calculation(self):
        """Test 6-month momentum calculation."""
        returns = pd.Series([0.02] * 10)
        
        # shift(2).rolling(5).sum() = 5 * 0.02 = 0.10
        mom6m = returns.shift(2).rolling(5).sum()
        
        assert abs(mom6m.iloc[7] - 0.10) < 0.001


class TestTrainTestSplit:
    """Tests for data splitting logic."""
    
    def test_no_data_leakage(self):
        """Test that train/val/test sets don't overlap."""
        dates = pd.Series([2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018])
        
        TRAIN_END = 2013
        VAL_END = 2016
        
        train_mask = dates <= TRAIN_END
        val_mask = (dates > TRAIN_END) & (dates <= VAL_END)
        test_mask = dates > VAL_END
        
        # No overlap
        assert not (train_mask & val_mask).any(), "Train and val overlap"
        assert not (train_mask & test_mask).any(), "Train and test overlap"
        assert not (val_mask & test_mask).any(), "Val and test overlap"
        
        # Complete coverage
        assert (train_mask | val_mask | test_mask).all(), "Some data not assigned"
    
    def test_correct_year_assignment(self):
        """Test that years are assigned to correct splits."""
        years = pd.Series([2010, 2013, 2014, 2016, 2017, 2020])
        
        TRAIN_END = 2013
        VAL_END = 2016
        
        train_mask = years <= TRAIN_END
        val_mask = (years > TRAIN_END) & (years <= VAL_END)
        test_mask = years > VAL_END
        
        # Check specific assignments
        assert train_mask.iloc[0] == True   # 2010 -> train
        assert train_mask.iloc[1] == True   # 2013 -> train
        assert val_mask.iloc[2] == True     # 2014 -> val
        assert val_mask.iloc[3] == True     # 2016 -> val
        assert test_mask.iloc[4] == True    # 2017 -> test
        assert test_mask.iloc[5] == True    # 2020 -> test


class TestReproducibility:
    """Tests for reproducibility utilities."""
    
    def test_seed_setting(self):
        """Test that setting seed produces same results."""
        set_seed(42)
        val1 = np.random.rand()
        tensor1 = torch.rand(1)
        
        set_seed(42)
        val2 = np.random.rand()
        tensor2 = torch.rand(1)
        
        assert val1 == val2, "NumPy random not reproducible"
        assert torch.equal(tensor1, tensor2), "PyTorch random not reproducible"
    
    def test_model_initialization_reproducible(self):
        """Test that model initialization is reproducible with same seed."""
        set_seed(42)
        model1 = AssetPricingNet(10)
        weights1 = model1.model[0].weight.clone()
        
        set_seed(42)
        model2 = AssetPricingNet(10)
        weights2 = model2.model[0].weight.clone()
        
        assert torch.allclose(weights1, weights2), "Model init not reproducible"


class TestDeviceSelection:
    """Tests for device selection."""
    
    def test_get_device_returns_valid_device(self):
        """Test that get_device returns a valid torch device."""
        device = get_device()
        
        assert isinstance(device, torch.device)
        assert device.type in ['cpu', 'cuda', 'mps']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

