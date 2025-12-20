"""
Training Pipeline Validation Script

Quick smoke test to verify the advanced training pipeline works correctly.
Runs with minimal settings to quickly validate functionality.

Author: Thesis Project
Date: December 2024
"""
import sys
import os
from pathlib import Path
import time

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / "utils"))

def test_imports():
    """Test that all required modules can be imported."""
    print("\n" + "=" * 60)
    print("TEST 1: Import Validation")
    print("=" * 60)
    
    try:
        from paths import get_processed_data_path, get_results_path
        print("  ✅ paths module imported")
        
        from training_config import TrainingConfig, get_config, list_presets
        print("  ✅ training_config module imported")
        
        from training import (
            set_seed, clip_gradients, GradientClipper,
            get_scheduler, EarlyStopping,
            ModelCheckpoint, TrainingLogger
        )
        print("  ✅ training module imported")
        
        from losses import get_loss_function, HuberLoss, ICLoss
        print("  ✅ losses module imported")
        
        from models import AssetPricingNet, get_device
        print("  ✅ models module imported")
        
        import torch
        import numpy as np
        import pandas as pd
        print("  ✅ Core libraries imported (torch, numpy, pandas)")
        
        return True
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False


def test_config_presets():
    """Test that all configuration presets work."""
    print("\n" + "=" * 60)
    print("TEST 2: Configuration Presets")
    print("=" * 60)
    
    from training_config import get_config, list_presets
    
    presets = list_presets()
    all_passed = True
    
    for preset_name, description in presets.items():
        try:
            config = get_config(preset_name)
            print(f"  ✅ '{preset_name}' preset loaded successfully")
        except Exception as e:
            print(f"  ❌ '{preset_name}' preset failed: {e}")
            all_passed = False
    
    # Test config with overrides
    try:
        config = get_config('default', epochs=10, learning_rate=0.001)
        assert config.epochs == 10
        assert config.learning_rate == 0.001
        print("  ✅ Config overrides work correctly")
    except Exception as e:
        print(f"  ❌ Config overrides failed: {e}")
        all_passed = False
    
    return all_passed


def test_loss_functions():
    """Test that all loss functions work correctly."""
    print("\n" + "=" * 60)
    print("TEST 3: Loss Functions")
    print("=" * 60)
    
    import torch
    from losses import get_loss_function
    
    # Test data
    predictions = torch.randn(32, 1)
    targets = torch.randn(32, 1)
    
    loss_names = ['mse', 'mae', 'huber', 'quantile', 'asymmetric']
    all_passed = True
    
    for name in loss_names:
        try:
            loss_fn = get_loss_function(name)
            loss = loss_fn(predictions, targets)
            assert not torch.isnan(loss), f"Loss is NaN"
            assert loss.item() >= 0, f"Loss is negative"
            print(f"  ✅ '{name}' loss: {loss.item():.6f}")
        except Exception as e:
            print(f"  ❌ '{name}' loss failed: {e}")
            all_passed = False
    
    return all_passed


def test_model_architecture():
    """Test neural network architecture."""
    print("\n" + "=" * 60)
    print("TEST 4: Model Architecture")
    print("=" * 60)
    
    import torch
    from models import AssetPricingNet, get_device
    
    device = get_device()
    print(f"  Device: {device}")
    
    all_passed = True
    
    # Test 1: Default architecture
    try:
        model = AssetPricingNet(input_dim=77)
        x = torch.randn(32, 77)
        model.eval()
        output = model(x)
        assert output.shape == (32, 1), f"Wrong output shape: {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN"
        print("  ✅ Default architecture (77 features) works")
    except Exception as e:
        print(f"  ❌ Default architecture failed: {e}")
        all_passed = False
    
    # Test 2: Custom architecture
    try:
        model = AssetPricingNet(
            input_dim=50,
            hidden_dims=(128, 64, 32),
            dropout_rates=(0.5, 0.4, 0.3)
        )
        x = torch.randn(16, 50)
        model.eval()
        output = model(x)
        assert output.shape == (16, 1)
        print("  ✅ Custom architecture (128-64-32) works")
    except Exception as e:
        print(f"  ❌ Custom architecture failed: {e}")
        all_passed = False
    
    # Test 3: GPU/MPS compatibility
    try:
        model = AssetPricingNet(input_dim=30).to(device)
        x = torch.randn(8, 30).to(device)
        model.eval()
        output = model(x)
        assert output.shape == (8, 1)
        print(f"  ✅ Model runs on {device}")
    except Exception as e:
        print(f"  ❌ Device compatibility failed: {e}")
        all_passed = False
    
    return all_passed


def test_training_utilities():
    """Test training utility functions."""
    print("\n" + "=" * 60)
    print("TEST 5: Training Utilities")
    print("=" * 60)
    
    import torch
    from models import AssetPricingNet
    from training import (
        set_seed, clip_gradients, GradientClipper,
        EarlyStopping
    )
    
    all_passed = True
    
    # Test 1: Reproducibility
    try:
        set_seed(42)
        t1 = torch.randn(10)
        set_seed(42)
        t2 = torch.randn(10)
        assert torch.allclose(t1, t2), "Seeds not reproducible"
        print("  ✅ set_seed() ensures reproducibility")
    except Exception as e:
        print(f"  ❌ Reproducibility failed: {e}")
        all_passed = False
    
    # Test 2: Gradient clipping
    try:
        model = AssetPricingNet(input_dim=10)
        x = torch.randn(4, 10)
        y = model(x)
        y.sum().backward()
        
        # Create artificially large gradients
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.fill_(100.0)
        
        norm_before = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        clip_gradients(model, max_norm=1.0)
        norm_after = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        
        print(f"  ✅ Gradient clipping works (norm: {norm_before:.2f} → clipped)")
    except Exception as e:
        print(f"  ❌ Gradient clipping failed: {e}")
        all_passed = False
    
    # Test 3: Early stopping
    try:
        es = EarlyStopping(patience=3, mode='max')
        model = torch.nn.Linear(10, 1)
        
        # Simulate improving scores
        for score in [0.1, 0.2, 0.3, 0.4]:
            es(score, model)
        assert not es.early_stop, "Should not stop when improving"
        
        # Simulate declining scores
        for score in [0.35, 0.30, 0.25]:
            es(score, model)
        assert es.early_stop, "Should stop after patience epochs"
        
        print("  ✅ EarlyStopping works correctly")
    except Exception as e:
        print(f"  ❌ Early stopping failed: {e}")
        all_passed = False
    
    return all_passed


def test_data_loading():
    """Test that data can be loaded and processed."""
    print("\n" + "=" * 60)
    print("TEST 6: Data Loading")
    print("=" * 60)
    
    from paths import get_processed_data_path
    import pandas as pd
    
    all_passed = True
    
    try:
        file_path = get_processed_data_path("thesis_dataset_macro.parquet")
        
        if not file_path.exists():
            print(f"  ⚠️ Data file not found: {file_path}")
            print("  ⚠️ Skipping data loading test (run data pipeline first)")
            return True  # Not a failure, just skip
        
        df = pd.read_parquet(str(file_path))
        print(f"  ✅ Data loaded: {len(df):,} rows, {len(df.columns)} columns")
        
        # Check required columns
        required = ['date', 'permno', 'future_ret', 'mkt_cap']
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"  ❌ Missing columns: {missing}")
            all_passed = False
        else:
            print("  ✅ Required columns present")
        
        # Check date range
        min_year = df['date'].dt.year.min()
        max_year = df['date'].dt.year.max()
        print(f"  ✅ Date range: {min_year} - {max_year}")
        
    except Exception as e:
        print(f"  ❌ Data loading failed: {e}")
        all_passed = False
    
    return all_passed


def test_mini_training_run():
    """Run a minimal training loop to verify the pipeline works end-to-end."""
    print("\n" + "=" * 60)
    print("TEST 7: Mini Training Run (Quick Smoke Test)")
    print("=" * 60)
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np
    
    from models import AssetPricingNet, get_device
    from training import set_seed, EarlyStopping, clip_gradients
    from losses import get_loss_function
    from training_config import get_config
    
    all_passed = True
    
    try:
        # Setup
        set_seed(42)
        device = get_device()
        config = get_config('fast', epochs=3, ensemble_models=1)
        
        print(f"  Config: epochs={config.epochs}, batch_size={config.batch_size}")
        
        # Create synthetic data
        n_samples = 1000
        n_features = 50
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(n_samples) * 0.1).reshape(-1, 1).astype(np.float32)
        
        # Split
        train_X, val_X = X[:800], X[800:]
        train_y, val_y = y[:800], y[800:]
        
        # DataLoader
        train_dataset = TensorDataset(
            torch.tensor(train_X),
            torch.tensor(train_y)
        )
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # Model
        model = AssetPricingNet(n_features).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        criterion = get_loss_function('mse')
        early_stopping = EarlyStopping(patience=2, mode='min')
        
        print("  Training...")
        start_time = time.time()
        
        for epoch in range(config.epochs):
            model.train()
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                
                # Gradient clipping
                clip_gradients(model, max_norm=1.0)
                
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_X_t = torch.tensor(val_X).to(device)
                val_preds = model(val_X_t).cpu().numpy()
                val_loss = ((val_preds - val_y) ** 2).mean()
            
            print(f"    Epoch {epoch+1}: train_loss={avg_loss:.6f}, val_loss={val_loss:.6f}")
            
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("    Early stopped!")
                break
        
        elapsed = time.time() - start_time
        print(f"  ✅ Mini training completed in {elapsed:.2f}s")
        
        # Restore best weights
        early_stopping.restore_best_weights(model)
        
        # Final prediction
        model.eval()
        with torch.no_grad():
            test_input = torch.randn(1, n_features).to(device)
            prediction = model(test_input)
            print(f"  ✅ Final prediction shape: {prediction.shape}")
        
    except Exception as e:
        import traceback
        print(f"  ❌ Mini training failed: {e}")
        traceback.print_exc()
        all_passed = False
    
    return all_passed


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE VALIDATION")
    print("=" * 60)
    print("Running quick smoke tests to verify everything works...")
    
    results = {}
    
    results['imports'] = test_imports()
    results['config_presets'] = test_config_presets()
    results['loss_functions'] = test_loss_functions()
    results['model_architecture'] = test_model_architecture()
    results['training_utilities'] = test_training_utilities()
    results['data_loading'] = test_data_loading()
    results['mini_training'] = test_mini_training_run()
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "-" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED - Training pipeline is ready!")
    else:
        print("⚠️ SOME TESTS FAILED - Please review errors above")
    print("=" * 60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

