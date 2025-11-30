#!/usr/bin/env python3
"""
Quick runner for Three-Layer Trend Explainability Analysis

Usage:
    python run_trend_explainability.py --model-path checkpoints/best_model.zip
"""

import sys
from pathlib import Path

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent))

from tools.trend_explainability import main

if __name__ == "__main__":
    # Default configuration for SmartFolio
    default_args = [
        "--market", "hs300",
        "--horizon", "1",
        "--relation-type", "hy",
        "--data-root", "dataset_default",
        "--device", "cpu",
        "--output-dir", "./explainability_results",
        "--tickers-csv", "tickers.csv",
        "--n-quantiles", "10",
        "--deterministic",
    ]
    
    # Parse user args and merge with defaults
    import argparse
    parser = argparse.ArgumentParser(description="Run trend explainability analysis")
    parser.add_argument("--model-path", required=True, help="Path to trained PPO model (.zip)")
    parser.add_argument("--test-start-date", required=True, help="Test start date (YYYY-MM-DD)")
    parser.add_argument("--test-end-date", required=True, help="Test end date (YYYY-MM-DD)")
    parser.add_argument("--market", default="hs300", help="Market code")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--output-dir", default="./explainability_results", help="Output directory")
    
    args, unknown = parser.parse_known_args()
    
    # Build final argv
    final_argv = [
        "--model-path", args.model_path,
        "--test-start-date", args.test_start_date,
        "--test-end-date", args.test_end_date,
        "--market", args.market,
        "--device", args.device,
        "--output-dir", args.output_dir,
        "--horizon", "1",
        "--relation-type", "hy",
        "--data-root", "dataset_default",
        "--tickers-csv", "tickers.csv",
        "--n-quantiles", "10",
        "--deterministic",
    ] + unknown
    
    print("\n" + "="*70)
    print("  THREE-LAYER TREND EXPLAINABILITY ANALYSIS")
    print("  Proving: TS-Momentum → LSTM-Trend → Policy-Logit → Weight")
    print("="*70 + "\n")
    
    main(final_argv)
    
    print("\n" + "="*70)
    print("  Analysis complete! Check explainability_results/ for outputs:")
    print("    - trend_explainability_results.json (full metrics)")
    print("    - trend_correlation_pyramid.png (causal chain)")
    print("    - trend_scatter_plots.png (correlations)")
    print("    - trend_quantile_uplift.png (uplift analysis)")
    print("    - trend_time_series_examples.png (stock examples)")
    print("    - trend_heatmap_per_stock.png (per-stock averages)")
    print("="*70 + "\n")
