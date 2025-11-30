#!/usr/bin/env python3
"""
Example: Test Three-Layer Trend Explainability

This script demonstrates how to run trend explainability analysis
on a trained SmartFolio model.
"""

import subprocess
import sys
from pathlib import Path

# Configuration
MODEL_PATH = "checkpoints/ppo_hgat_custom_20251128_015405.zip"
TEST_START = "2024-01-01"  # Adjust based on your dataset
TEST_END = "2024-12-31"
MARKET = "hs300"
DEVICE = "cpu"  # Change to "cuda" if GPU available

def main():
    """Run trend explainability example."""
    
    print("\n" + "="*70)
    print("  TESTING THREE-LAYER TREND EXPLAINABILITY")
    print("="*70)
    
    # Check if model exists
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"\n❌ Error: Model not found at {MODEL_PATH}")
        print("   Please update MODEL_PATH in this script to point to your trained model.")
        return 1
    
    print(f"\n✓ Found model: {MODEL_PATH}")
    print(f"✓ Test period: {TEST_START} to {TEST_END}")
    print(f"✓ Market: {MARKET}")
    print(f"✓ Device: {DEVICE}")
    
    # Build command
    cmd = [
        sys.executable,
        "run_trend_explainability.py",
        "--model-path", MODEL_PATH,
        "--test-start-date", TEST_START,
        "--test-end-date", TEST_END,
        "--market", MARKET,
        "--device", DEVICE,
    ]
    
    print("\n" + "-"*70)
    print("Running command:")
    print(" ".join(cmd))
    print("-"*70 + "\n")
    
    # Run analysis
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "="*70)
        print("  ✅ SUCCESS! Analysis complete.")
        print("="*70)
        print("\nGenerated files in explainability_results/:")
        print("  📊 trend_correlation_pyramid.png")
        print("  📊 trend_scatter_plots.png")
        print("  📊 trend_quantile_uplift.png")
        print("  📊 trend_time_series_examples.png")
        print("  📊 trend_heatmap_per_stock.png")
        print("  📄 trend_explainability_results.json")
        print("\n" + "="*70)
        return 0
    except subprocess.CalledProcessError as e:
        print("\n" + "="*70)
        print(f"  ❌ ERROR: Analysis failed with code {e.returncode}")
        print("="*70)
        print("\nCommon issues:")
        print("  1. Test date range doesn't match dataset")
        print("  2. Dataset directory not found")
        print("  3. Model architecture mismatch")
        print("\nCheck the error messages above for details.")
        return 1
    except FileNotFoundError:
        print("\n❌ Error: Python not found or run_trend_explainability.py missing")
        return 1

if __name__ == "__main__":
    sys.exit(main())
