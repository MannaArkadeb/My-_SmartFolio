#!/usr/bin/env python3
"""
Unit tests for trend explainability functions.

Run with: python -m pytest test_trend_explainability.py
Or simply: python test_trend_explainability.py
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.trend_explainability import (
    compute_ts_momentum,
    TrendData,
)


def test_compute_ts_momentum():
    """Test TS-Momentum calculation."""
    # Create synthetic data: uptrend, downtrend, flat
    N, L, D = 3, 20, 1
    ts_data = np.zeros((N, L, D))
    
    # Stock 0: Uptrend (0 to 1)
    ts_data[0, :, 0] = np.linspace(0, 1, L)
    
    # Stock 1: Downtrend (1 to 0)
    ts_data[1, :, 0] = np.linspace(1, 0, L)
    
    # Stock 2: Flat (0.5 everywhere)
    ts_data[2, :, 0] = 0.5
    
    momentum = compute_ts_momentum(ts_data, return_channel=0)
    
    # Check results
    assert momentum[0] > 0, f"Uptrend should have positive momentum, got {momentum[0]}"
    assert momentum[1] < 0, f"Downtrend should have negative momentum, got {momentum[1]}"
    assert abs(momentum[2]) < 0.01, f"Flat should have near-zero momentum, got {momentum[2]}"
    
    print("✅ test_compute_ts_momentum passed")


def test_trend_data_validation():
    """Test TrendData validation."""
    T, N = 10, 5
    
    # Valid data
    valid_data = TrendData(
        ts_momentum=np.random.randn(T, N),
        lstm_trend=np.random.randn(T, N),
        policy_logit=np.random.randn(T, N),
        portfolio_weight=np.random.rand(T, N),
        decision_steps=list(range(T)),
        stock_names=[f"Stock_{i}" for i in range(N)],
    )
    assert valid_data.ts_momentum.shape == (T, N)
    
    # Invalid: mismatched shapes
    try:
        invalid_data = TrendData(
            ts_momentum=np.random.randn(T, N),
            lstm_trend=np.random.randn(T, N+1),  # Wrong shape!
            policy_logit=np.random.randn(T, N),
            portfolio_weight=np.random.rand(T, N),
            decision_steps=list(range(T)),
            stock_names=[f"Stock_{i}" for i in range(N)],
        )
        assert False, "Should have raised ValueError for mismatched shapes"
    except ValueError as e:
        assert "Mismatched shapes" in str(e)
    
    print("✅ test_trend_data_validation passed")


def test_correlation_computation():
    """Test that correlations work as expected."""
    from scipy import stats
    
    # Create correlated data
    np.random.seed(42)  # For reproducibility
    N = 100
    x = np.random.randn(N)
    y = 0.8 * x + 0.2 * np.random.randn(N)  # Should have ~0.8 correlation
    
    corr_pearson = np.corrcoef(x, y)[0, 1]
    corr_spearman = stats.spearmanr(x, y)[0]
    
    # More relaxed bounds since correlation can vary
    assert 0.5 < corr_pearson < 1.0, f"Unexpected Pearson correlation: {corr_pearson}"
    assert 0.5 < corr_spearman < 1.0, f"Unexpected Spearman correlation: {corr_spearman}"
    assert corr_pearson > 0, "Pearson correlation should be positive"
    assert corr_spearman > 0, "Spearman correlation should be positive"
    
    print("✅ test_correlation_computation passed")


def test_quantile_logic():
    """Test quantile binning logic."""
    # Create data with clear trend
    data = np.arange(100).astype(float)  # 0 to 99
    
    # Split into 10 quantiles
    n_quantiles = 10
    quantiles = np.linspace(0, 100, n_quantiles + 1)
    quantile_edges = np.percentile(data, quantiles)
    quantile_indices = np.digitize(data, quantile_edges[1:-1])
    
    # Check that bottom quantile has low values
    bottom_mask = (quantile_indices == 0)
    top_mask = (quantile_indices == n_quantiles - 1)
    
    bottom_mean = data[bottom_mask].mean()
    top_mean = data[top_mask].mean()
    
    assert bottom_mean < 20, f"Bottom quantile mean too high: {bottom_mean}"
    assert top_mean > 80, f"Top quantile mean too low: {top_mean}"
    assert top_mean > bottom_mean, "Top should be higher than bottom"
    
    print("✅ test_quantile_logic passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("  Running Trend Explainability Unit Tests")
    print("="*60 + "\n")
    
    tests = [
        test_compute_ts_momentum,
        test_trend_data_validation,
        test_correlation_computation,
        test_quantile_logic,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
