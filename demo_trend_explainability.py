#!/usr/bin/env python3
"""
Demo: Three-Layer Trend Explainability with Synthetic Data

This script demonstrates the trend explainability system using synthetic data
when real dataset is not available. It generates mock time-series with clear
trends and shows how the analysis works.
"""

import numpy as np
import torch
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))

from tools.trend_explainability import (
    TrendData,
    compute_ts_momentum,
    analyze_correlation_pyramid,
    analyze_quantile_uplift,
    analyze_path_regressions,
    analyze_per_stock_slopes,
    create_visualizations,
)


def generate_synthetic_trend_data(n_decisions=50, n_stocks=30, seed=42, 
                                 momentum_strength=0.7, lstm_sensitivity=0.75,
                                 policy_amplification=0.85, noise_level=0.3):
    """
    Generate synthetic trend data with realistic market-like distributions.
    
    Instead of artificial buckets, generates continuous distributions similar to
    real market data where momentum follows a normal-like distribution.
    
    Args:
        n_decisions: Number of rebalancing decisions
        n_stocks: Number of stocks
        seed: Random seed for reproducibility
        momentum_strength: Controls persistence of momentum (0-1, higher = more persistent)
        lstm_sensitivity: Correlation strength TS→LSTM (0-1)
        policy_amplification: Correlation strength LSTM→Logit (0-1)
        noise_level: Amount of noise at each layer (0-1, higher = more noise)
    """
    np.random.seed(seed)
    print("\n🔧 Generating synthetic trend data...")
    print(f"   Decisions: {n_decisions}, Stocks: {n_stocks}")
    print(f"   Parameters: momentum_strength={momentum_strength:.2f}, "
          f"lstm_sensitivity={lstm_sensitivity:.2f}, "
          f"policy_amplification={policy_amplification:.2f}")
    
    # Sample base momentum from realistic distribution (normal with fat tails)
    # Real markets: most stocks cluster near zero, some strong winners/losers
    base_momentum = np.random.standard_t(df=5, size=n_stocks) * 0.4  # t-distribution for fat tails
    
    # Generate time series with temporal autocorrelation
    ts_momentum = np.zeros((n_decisions, n_stocks))
    lstm_trend = np.zeros((n_decisions, n_stocks))
    policy_logit = np.zeros((n_decisions, n_stocks))
    portfolio_weight = np.zeros((n_decisions, n_stocks))
    
    # Initialize with base momentum
    ts_momentum[0] = base_momentum + np.random.randn(n_stocks) * noise_level
    
    for t in range(n_decisions):
        if t > 0:
            # Layer 1: TS-Momentum with autocorrelation (trends persist over time)
            # AR(1) process: x_t = ρ*x_{t-1} + ε
            ts_momentum[t] = (momentum_strength * ts_momentum[t-1] + 
                             (1 - momentum_strength) * base_momentum +
                             np.random.randn(n_stocks) * noise_level)
        
        # Layer 2: LSTM-Trend (LSTM learns to track momentum with some noise)
        # Linear relationship with controlled correlation
        lstm_signal = ts_momentum[t]
        lstm_noise = np.random.randn(n_stocks) * noise_level * np.sqrt(1 - lstm_sensitivity**2)
        lstm_trend[t] = 2.0 * lstm_sensitivity * lstm_signal + lstm_noise
        
        # Layer 3: Policy-Logit (Policy uses LSTM with additional processing)
        # Amplifies signal with some independent noise
        policy_signal = lstm_trend[t]
        policy_noise = np.random.randn(n_stocks) * noise_level * np.sqrt(1 - policy_amplification**2)
        policy_logit[t] = 1.5 * policy_amplification * policy_signal + policy_noise
        
        # Final: Portfolio-Weight (Softmax + realistic constraints)
        # Temperature-scaled softmax (realistic portfolio behavior)
        temperature = 2.0 + 0.5 * np.random.randn()  # Varies slightly over time
        raw_weights = np.exp(policy_logit[t] / temperature)
        raw_weights = raw_weights / raw_weights.sum()
        
        # Apply realistic portfolio constraints (max weight, min weight)
        max_weight = 0.10 + 0.05 * np.random.rand()  # Varies: 10-15%
        min_weight = 0.001
        
        # Iterative constraint application (like real risk management)
        for _ in range(10):
            raw_weights = np.clip(raw_weights, min_weight, max_weight)
            weight_sum = raw_weights.sum()
            if weight_sum > 0:
                raw_weights = raw_weights / weight_sum
            else:
                raw_weights = np.ones(n_stocks) / n_stocks
                break
        
        portfolio_weight[t] = raw_weights
    
    # Generate realistic stock names (ticker-like)
    stock_names = [f"STK{i:03d}" for i in range(n_stocks)]
    
    # Simulate monthly rebalancing (21 trading days)
    decision_steps = list(range(0, n_decisions * 21, 21))
    
    # Calculate statistics for reporting
    avg_momentum = ts_momentum.mean(axis=0)
    n_positive = (avg_momentum > 0.1).sum()
    n_negative = (avg_momentum < -0.1).sum()
    n_neutral = n_sto
    cks - n_positive - n_negative
    
    print("   ✓ Generated with realistic market-like distributions")
    print(f"   - Positive momentum stocks: {n_positive} ({100*n_positive/n_stocks:.1f}%)")
    print(f"   - Negative momentum stocks: {n_negative} ({100*n_negative/n_stocks:.1f}%)")
    print(f"   - Neutral stocks: {n_neutral} ({100*n_neutral/n_stocks:.1f}%)")
    print(f"   - Momentum range: [{ts_momentum.min():.3f}, {ts_momentum.max():.3f}]")
    
    return TrendData(
        ts_momentum=ts_momentum,
        lstm_trend=lstm_trend,
        policy_logit=policy_logit,
        portfolio_weight=portfolio_weight,
        decision_steps=decision_steps,
        stock_names=stock_names,
    )


def main():
    """Run demo analysis with synthetic data."""
    
    print("\n" + "="*70)
    print("  THREE-LAYER TREND EXPLAINABILITY - SYNTHETIC DEMO")
    print("  (Using mock data to demonstrate the system)")
    print("="*70)
    
    # Generate synthetic data with realistic parameters
    # You can adjust these to simulate different market conditions:
    # - momentum_strength: how persistent trends are (0.7 = fairly persistent)
    # - lstm_sensitivity: how well LSTM tracks momentum (0.75 = good tracking)
    # - policy_amplification: how much policy follows LSTM (0.85 = strong following)
    # - noise_level: market noise/randomness (0.3 = moderate noise)
    data = generate_synthetic_trend_data(
        n_decisions=50, 
        n_stocks=30, 
        seed=42,
        momentum_strength=0.7,      # Trend persistence
        lstm_sensitivity=0.75,       # TS→LSTM correlation strength
        policy_amplification=0.85,   # LSTM→Logit correlation strength
        noise_level=0.3,            # Overall noise level
    )
    
    # Run analyses
    print("\n" + "="*70)
    print("  RUNNING ANALYSES")
    print("="*70)
    
    analyses = {
        "correlation_pyramid": analyze_correlation_pyramid(data),
        "quantile_uplift": analyze_quantile_uplift(data, n_quantiles=10),
        "path_regressions": analyze_path_regressions(data),
        "per_stock_slopes": analyze_per_stock_slopes(data),
    }
    
    # Create output directory
    output_dir = Path("explainability_results_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    print("\n" + "="*70)
    print("  GENERATING VISUALIZATIONS")
    print("="*70)
    
    create_visualizations(data, analyses, output_dir)
    
    # Save results
    print("\n" + "="*70)
    print("  SAVING RESULTS")
    print("="*70)
    
    results = {
        "metadata": {
            "type": "synthetic_demo",
            "num_stocks": len(data.stock_names),
            "num_decisions": len(data.decision_steps),
            "stock_names": data.stock_names,
        },
        "analyses": analyses,
        "summary": {
            "conclusion": "Strong causal chain validated (synthetic data)" if all([
                analyses["correlation_pyramid"]["TS_to_LSTM"]["spearman"] > 0.3,
                analyses["correlation_pyramid"]["LSTM_to_Logit"]["spearman"] > 0.3,
                analyses["correlation_pyramid"]["Logit_to_Weight"]["spearman"] > 0.3,
            ]) else "Weak causal chain",
            "key_findings": [
                f"TS-Momentum correlates with LSTM-Trend (ρ={analyses['correlation_pyramid']['TS_to_LSTM']['spearman']:.3f})",
                f"LSTM-Trend drives Policy-Logit (ρ={analyses['correlation_pyramid']['LSTM_to_Logit']['spearman']:.3f})",
                f"Policy-Logit determines Weight (ρ={analyses['correlation_pyramid']['Logit_to_Weight']['spearman']:.3f})",
                f"Top-quantile TS-Momentum receives {analyses['quantile_uplift']['TS_Momentum']['uplift_ratio']:.2f}x more weight",
            ],
            "note": "This demo uses synthetic data to demonstrate the analysis. Run with real dataset for actual portfolio insights.",
        },
    }
    
    output_path = output_dir / "trend_explainability_results_demo.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"  ✓ Saved: {output_path}")
    
    # Final summary
    print("\n" + "="*70)
    print("  ✅ DEMO COMPLETE!")
    print("="*70)
    print(f"\nGenerated files in {output_dir}/:")
    print("  📊 trend_correlation_pyramid.png")
    print("  📊 trend_scatter_plots.png")
    print("  📊 trend_quantile_uplift.png")
    print("  📊 trend_time_series_examples.png")
    print("  📊 trend_heatmap_per_stock.png")
    print("  📄 trend_explainability_results_demo.json")
    
    print("\n" + "="*70)
    print("  KEY FINDINGS (Synthetic Data)")
    print("="*70)
    for finding in results["summary"]["key_findings"]:
        print(f"  • {finding}")
    
    print("\n" + "="*70)
    print("  NEXT STEPS")
    print("="*70)
    print("  1. Review visualizations in explainability_results_demo/")
    print("  2. To run on real data:")
    print("     - Ensure dataset is in dataset_default/")
    print("     - Update test dates to match your data")
    print("     - Run: python run_trend_explainability.py \\")
    print("            --model-path YOUR_MODEL.zip \\")
    print("            --test-start-date YYYY-MM-DD \\")
    print("            --test-end-date YYYY-MM-DD")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
