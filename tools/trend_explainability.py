#!/usr/bin/env python3
"""
Three-Layer Trend Explainability for SmartFolio

This module proves the causal chain:
  TS-Momentum → LSTM-Trend → Policy-Logit → Portfolio-Weight

Analyzes:
- Layer 1 (TS-Momentum): Simple time-series momentum from raw returns
- Layer 2 (LSTM-Trend): Temporal-only score from LSTM embeddings
- Layer 3 (Policy-Logit): PPO logits before env constraints
- Final (Portfolio-Weight): Actual allocations after softmax + risk constraints

Outputs:
- Correlation pyramid (Pearson & Spearman)
- Quantile uplift analysis
- Path regressions
- Per-stock slope distributions
- Comprehensive visualizations (scatter, heatmaps, time-series)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import LinearRegression
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.data_loader import AllGraphDataSampler
from env.portfolio_env import StockPortfolioEnv
from stable_baselines3 import PPO


@dataclass
class TrendData:
    """Container for all three layers of trend signals plus weights."""
    ts_momentum: np.ndarray  # [T, N] - Time-series momentum
    lstm_trend: np.ndarray   # [T, N] - LSTM temporal-only score
    policy_logit: np.ndarray # [T, N] - PPO logits
    portfolio_weight: np.ndarray  # [T, N] - Final env weights
    decision_steps: List[int]  # Decision step indices
    stock_names: List[str]  # Ticker names
    
    def __post_init__(self):
        """Validate shapes."""
        shapes = [
            self.ts_momentum.shape,
            self.lstm_trend.shape,
            self.policy_logit.shape,
            self.portfolio_weight.shape,
        ]
        if len(set(shapes)) != 1:
            raise ValueError(f"Mismatched shapes: {shapes}")
        
        T, N = self.ts_momentum.shape
        if len(self.decision_steps) != T:
            raise ValueError(f"decision_steps length {len(self.decision_steps)} != T={T}")
        if len(self.stock_names) != N:
            raise ValueError(f"stock_names length {len(self.stock_names)} != N={N}")


def compute_ts_momentum(ts_data: np.ndarray, return_channel: int = 0) -> np.ndarray:
    """
    Compute time-series momentum from raw TS data.
    
    Args:
        ts_data: [N, L, D] - per-stock time series (N stocks, L lookback, D features)
        return_channel: which feature channel represents returns/price
    
    Returns:
        momentum: [N] - scalar momentum per stock (late-period mean - early-period mean)
    """
    N, L, D = ts_data.shape
    if return_channel >= D:
        raise ValueError(f"return_channel {return_channel} >= feature dim {D}")
    
    returns = ts_data[:, :, return_channel]  # [N, L]
    
    # Split into early and late halves
    mid = L // 2
    early_mean = returns[:, :mid].mean(axis=1)  # [N]
    late_mean = returns[:, mid:].mean(axis=1)   # [N]
    
    momentum = late_mean - early_mean  # [N]
    return momentum


def extract_lstm_embeddings(policy_net, features_tensor: torch.Tensor) -> torch.Tensor:
    """
    Extract LSTM node embeddings (before graph attention).
    
    Args:
        policy_net: TemporalHGAT model
        features_tensor: [B, obs_dim] observation tensor
    
    Returns:
        node_embeddings: [B, N, H] - LSTM embeddings per stock
    """
    batch = features_tensor.shape[0]
    num_stocks = policy_net.num_stocks
    lookback = policy_net.lookback
    input_dim = policy_net.input_dim
    
    adj_size = num_stocks * num_stocks
    ts_size = num_stocks * lookback * input_dim
    
    # Parse observation (same layout as model forward)
    ptr = 3 * adj_size  # Skip ind, pos, neg
    ts_flat = features_tensor[:, ptr:ptr + ts_size]
    
    # Reshape for LSTM: [B*N, L, D]
    ts_features = ts_flat.reshape(batch * num_stocks, lookback, input_dim)
    
    # Run LSTM
    with torch.no_grad():
        _, (h_n, _) = policy_net.lstm(ts_features)
        node_embeddings = h_n[-1].reshape(batch, num_stocks, -1)  # [B, N, H]
    
    return node_embeddings


def compute_lstm_trend(policy_net, features_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute temporal-only trend score from LSTM embeddings.
    
    Uses the same output head that produces final logits, but applied
    only to LSTM embeddings (before graph attention fusion).
    
    Args:
        policy_net: TemporalHGAT model
        features_tensor: [B, obs_dim] observation tensor
    
    Returns:
        lstm_trend: [B, N] - scalar trend score per stock
    """
    node_embeddings = extract_lstm_embeddings(policy_net, features_tensor)  # [B, N, H]
    
    # Apply output head (same as used in full forward, but on temporal features only)
    with torch.no_grad():
        lstm_trend = policy_net.output_head(node_embeddings).squeeze(-1)  # [B, N]
    
    return lstm_trend


def collect_trend_data(
    loader: DataLoader,
    model: PPO,
    args: argparse.Namespace,
    device: torch.device,
    stock_names: List[str],
) -> TrendData:
    """
    Collect all three layers of trend signals across the test period.
    
    Returns:
        TrendData with shapes [T_decisions, N] where T_decisions = number of rebalancing steps
    """
    ts_momentum_list = []
    lstm_trend_list = []
    policy_logit_list = []
    portfolio_weight_list = []
    decision_step_list = []
    
    print("\n=== Collecting Three-Layer Trend Data ===")
    
    for batch_idx, batch in enumerate(loader):
        print(f"Processing batch {batch_idx + 1}/{len(loader)}")
        
        # Process data (same as attention_viz)
        corr = batch["corr"].to(device).squeeze()
        ts_features = batch["ts_features"].to(device).squeeze()
        features = batch["features"].to(device).squeeze()
        ind = batch["industry_matrix"].to(device).squeeze()
        pos = batch["pos_matrix"].to(device).squeeze()
        neg = batch["neg_matrix"].to(device).squeeze()
        returns = batch["labels"].to(device).squeeze()
        pyg_data = batch["pyg_data"].to(device)
        
        # Create environment
        env = StockPortfolioEnv(
            args=args,
            corr=corr,
            ts_features=ts_features,
            features=features,
            ind=ind,
            pos=pos,
            neg=neg,
            returns=returns,
            pyg_data=pyg_data,
            mode="test",
            ind_yn=args.ind_yn,
            pos_yn=args.pos_yn,
            neg_yn=args.neg_yn,
            reward_net=None,
            device=str(device),
        )
        env.seed(args.seed)
        vec_env, obs = env.get_sb_env()
        vec_env.reset()
        
        # Access underlying env for weights
        underlying_env = vec_env.envs[0]
        
        max_steps = returns.shape[0] if hasattr(returns, "shape") else len(returns)
        step_limit = args.max_steps if args.max_steps is not None else max_steps
        
        for step in range(int(step_limit)):
            # Check if this is a rebalancing step
            is_rebalancing = (underlying_env.current_step % underlying_env.rebalance_window == 0) or (underlying_env.current_step == 1)
            
            if is_rebalancing:
                # Get current TS data for momentum calculation
                if torch.is_tensor(ts_features):
                    ts_data = ts_features[underlying_env.current_step].cpu().numpy()
                else:
                    ts_data = ts_features[underlying_env.current_step]
                
                # Layer 1: TS-Momentum
                ts_mom = compute_ts_momentum(ts_data, return_channel=0)
                
                # Get policy features for LSTM and logits
                obs_tensor, _ = model.policy.obs_to_tensor(obs)
                features_tensor = model.policy.extract_features(obs_tensor.to(device))
                
                # Layer 2: LSTM-Trend (temporal-only score)
                lstm_tr = compute_lstm_trend(
                    model.policy.mlp_extractor.policy_net,
                    features_tensor
                ).detach().cpu().numpy()[0]
                
                # Layer 3: Policy-Logit
                with torch.no_grad():
                    policy_logits = model.policy.mlp_extractor.policy_net(
                        features_tensor,
                        require_weights=False
                    ).detach().cpu().numpy()[0]
                
                # Record before step
                ts_momentum_list.append(ts_mom)
                lstm_trend_list.append(lstm_tr)
                policy_logit_list.append(policy_logits)
                decision_step_list.append(underlying_env.current_step)
            
            # Take action
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, rewards, dones, _info = vec_env.step(action)
            
            # Record final weight after step (if was rebalancing)
            if is_rebalancing:
                final_weights = underlying_env.weights_history[-1]
                portfolio_weight_list.append(final_weights.copy())
            
            if dones[0]:
                break
        
        vec_env.close()
    
    print(f"\nCollected {len(decision_step_list)} rebalancing decision points")
    
    return TrendData(
        ts_momentum=np.stack(ts_momentum_list, axis=0),
        lstm_trend=np.stack(lstm_trend_list, axis=0),
        policy_logit=np.stack(policy_logit_list, axis=0),
        portfolio_weight=np.stack(portfolio_weight_list, axis=0),
        decision_steps=decision_step_list,
        stock_names=stock_names,
    )


def analyze_correlation_pyramid(data: TrendData) -> Dict:
    """
    Compute correlation pyramid: TS→LSTM, LSTM→Logit, Logit→Weight.
    
    Returns dict with Pearson and Spearman correlations.
    """
    print("\n=== Correlation Pyramid Analysis ===")
    
    # Flatten over time and stocks for global correlations
    ts_flat = data.ts_momentum.flatten()
    lstm_flat = data.lstm_trend.flatten()
    logit_flat = data.policy_logit.flatten()
    weight_flat = data.portfolio_weight.flatten()
    
    results = {
        "TS_to_LSTM": {
            "pearson": float(np.corrcoef(ts_flat, lstm_flat)[0, 1]),
            "spearman": float(stats.spearmanr(ts_flat, lstm_flat)[0]),
        },
        "LSTM_to_Logit": {
            "pearson": float(np.corrcoef(lstm_flat, logit_flat)[0, 1]),
            "spearman": float(stats.spearmanr(lstm_flat, logit_flat)[0]),
        },
        "Logit_to_Weight": {
            "pearson": float(np.corrcoef(logit_flat, weight_flat)[0, 1]),
            "spearman": float(stats.spearmanr(logit_flat, weight_flat)[0]),
        },
        "TS_to_Weight": {
            "pearson": float(np.corrcoef(ts_flat, weight_flat)[0, 1]),
            "spearman": float(stats.spearmanr(ts_flat, weight_flat)[0]),
        },
    }
    
    # Cross-sectional correlations (per timestep)
    T = data.ts_momentum.shape[0]
    cs_corr_ts_lstm = []
    cs_corr_lstm_logit = []
    cs_corr_logit_weight = []
    
    for t in range(T):
        cs_corr_ts_lstm.append(stats.spearmanr(data.ts_momentum[t], data.lstm_trend[t])[0])
        cs_corr_lstm_logit.append(stats.spearmanr(data.lstm_trend[t], data.policy_logit[t])[0])
        cs_corr_logit_weight.append(stats.spearmanr(data.policy_logit[t], data.portfolio_weight[t])[0])
    
    results["cross_sectional"] = {
        "TS_to_LSTM_mean": float(np.mean(cs_corr_ts_lstm)),
        "TS_to_LSTM_std": float(np.std(cs_corr_ts_lstm)),
        "LSTM_to_Logit_mean": float(np.mean(cs_corr_lstm_logit)),
        "LSTM_to_Logit_std": float(np.std(cs_corr_lstm_logit)),
        "Logit_to_Weight_mean": float(np.mean(cs_corr_logit_weight)),
        "Logit_to_Weight_std": float(np.std(cs_corr_logit_weight)),
    }
    
    print("\nGlobal Correlations (all decisions):")
    print(f"  TS-Momentum → LSTM-Trend:  Pearson={results['TS_to_LSTM']['pearson']:.4f}, Spearman={results['TS_to_LSTM']['spearman']:.4f}")
    print(f"  LSTM-Trend → Policy-Logit: Pearson={results['LSTM_to_Logit']['pearson']:.4f}, Spearman={results['LSTM_to_Logit']['spearman']:.4f}")
    print(f"  Policy-Logit → Weight:     Pearson={results['Logit_to_Weight']['pearson']:.4f}, Spearman={results['Logit_to_Weight']['spearman']:.4f}")
    print(f"  TS-Momentum → Weight:      Pearson={results['TS_to_Weight']['pearson']:.4f}, Spearman={results['TS_to_Weight']['spearman']:.4f}")
    
    print("\nCross-Sectional Correlations (mean ± std):")
    print(f"  TS→LSTM:  {results['cross_sectional']['TS_to_LSTM_mean']:.4f} ± {results['cross_sectional']['TS_to_LSTM_std']:.4f}")
    print(f"  LSTM→Logit: {results['cross_sectional']['LSTM_to_Logit_mean']:.4f} ± {results['cross_sectional']['LSTM_to_Logit_std']:.4f}")
    print(f"  Logit→Weight: {results['cross_sectional']['Logit_to_Weight_mean']:.4f} ± {results['cross_sectional']['Logit_to_Weight_std']:.4f}")
    
    return results


def analyze_quantile_uplift(data: TrendData, n_quantiles: int = 10) -> Dict:
    """
    Compute quantile uplift: for each quantile of each trend layer,
    compute average values of subsequent layers and final weight.
    """
    print(f"\n=== Quantile Uplift Analysis (n={n_quantiles}) ===")
    
    results = {}
    
    # For each trend layer, compute quantile → next layer means
    for layer_name, layer_data in [
        ("TS_Momentum", data.ts_momentum),
        ("LSTM_Trend", data.lstm_trend),
        ("Policy_Logit", data.policy_logit),
    ]:
        flat_trend = layer_data.flatten()
        
        # Compute quantiles
        quantiles = np.linspace(0, 100, n_quantiles + 1)
        quantile_edges = np.percentile(flat_trend, quantiles)
        
        # Assign each observation to a quantile
        quantile_indices = np.digitize(flat_trend, quantile_edges[1:-1])
        
        # Compute mean weight per quantile
        mean_weights = []
        quantile_labels = []
        for q in range(n_quantiles):
            mask = (quantile_indices == q)
            if mask.sum() > 0:
                mean_weight = data.portfolio_weight.flatten()[mask].mean()
                mean_weights.append(float(mean_weight))
                quantile_labels.append(f"Q{q+1}")
            else:
                mean_weights.append(0.0)
                quantile_labels.append(f"Q{q+1}")
        
        results[layer_name] = {
            "quantile_labels": quantile_labels,
            "mean_weights": mean_weights,
            "uplift_ratio": float(mean_weights[-1] / mean_weights[0]) if mean_weights[0] > 1e-6 else 0.0,
        }
        
        print(f"\n{layer_name} → Portfolio Weight:")
        print(f"  Bottom quantile (Q1) avg weight: {mean_weights[0]:.6f}")
        print(f"  Top quantile (Q{n_quantiles}) avg weight: {mean_weights[-1]:.6f}")
        print(f"  Uplift ratio (top/bottom): {results[layer_name]['uplift_ratio']:.2f}x")
    
    return results


def analyze_path_regressions(data: TrendData) -> Dict:
    """
    Fit linear regressions along the causal chain.
    """
    print("\n=== Path Regression Analysis ===")
    
    # Flatten data
    ts_flat = data.ts_momentum.flatten().reshape(-1, 1)
    lstm_flat = data.lstm_trend.flatten().reshape(-1, 1)
    logit_flat = data.policy_logit.flatten().reshape(-1, 1)
    weight_flat = data.portfolio_weight.flatten()
    
    # Regression 1: LSTM ~ TS
    reg1 = LinearRegression().fit(ts_flat, data.lstm_trend.flatten())
    
    # Regression 2: Logit ~ LSTM
    reg2 = LinearRegression().fit(lstm_flat, data.policy_logit.flatten())
    
    # Regression 3: Weight ~ Logit
    reg3 = LinearRegression().fit(logit_flat, weight_flat)
    
    # Chained regression: Weight ~ TS + LSTM + Logit
    X_all = np.hstack([ts_flat, lstm_flat, logit_flat])
    reg_chain = LinearRegression().fit(X_all, weight_flat)
    
    results = {
        "TS_to_LSTM": {
            "slope": float(reg1.coef_[0]),
            "intercept": float(reg1.intercept_),
            "r2": float(reg1.score(ts_flat, data.lstm_trend.flatten())),
        },
        "LSTM_to_Logit": {
            "slope": float(reg2.coef_[0]),
            "intercept": float(reg2.intercept_),
            "r2": float(reg2.score(lstm_flat, data.policy_logit.flatten())),
        },
        "Logit_to_Weight": {
            "slope": float(reg3.coef_[0]),
            "intercept": float(reg3.intercept_),
            "r2": float(reg3.score(logit_flat, weight_flat)),
        },
        "Chained_Regression": {
            "coef_TS": float(reg_chain.coef_[0]),
            "coef_LSTM": float(reg_chain.coef_[1]),
            "coef_Logit": float(reg_chain.coef_[2]),
            "intercept": float(reg_chain.intercept_),
            "r2": float(reg_chain.score(X_all, weight_flat)),
        },
    }
    
    print("\nLinear Regressions:")
    print(f"  TS → LSTM:      slope={results['TS_to_LSTM']['slope']:.4f}, R²={results['TS_to_LSTM']['r2']:.4f}")
    print(f"  LSTM → Logit:   slope={results['LSTM_to_Logit']['slope']:.4f}, R²={results['LSTM_to_Logit']['r2']:.4f}")
    print(f"  Logit → Weight: slope={results['Logit_to_Weight']['slope']:.4f}, R²={results['Logit_to_Weight']['r2']:.4f}")
    print(f"\nChained Regression (Weight ~ TS + LSTM + Logit):")
    print(f"  Coef TS:    {results['Chained_Regression']['coef_TS']:.6f}")
    print(f"  Coef LSTM:  {results['Chained_Regression']['coef_LSTM']:.6f}")
    print(f"  Coef Logit: {results['Chained_Regression']['coef_Logit']:.6f}")
    print(f"  R²: {results['Chained_Regression']['r2']:.4f}")
    
    return results


def analyze_per_stock_slopes(data: TrendData) -> Dict:
    """
    For each stock, fit regressions across time and report slope distributions.
    """
    print("\n=== Per-Stock Slope Analysis ===")
    
    N = data.ts_momentum.shape[1]
    
    slopes_ts_lstm = []
    slopes_lstm_logit = []
    slopes_logit_weight = []
    
    for i in range(N):
        # TS → LSTM
        X = data.ts_momentum[:, i].reshape(-1, 1)
        y = data.lstm_trend[:, i]
        if np.std(X) > 1e-6 and np.std(y) > 1e-6:
            reg = LinearRegression().fit(X, y)
            slopes_ts_lstm.append(reg.coef_[0])
        
        # LSTM → Logit
        X = data.lstm_trend[:, i].reshape(-1, 1)
        y = data.policy_logit[:, i]
        if np.std(X) > 1e-6 and np.std(y) > 1e-6:
            reg = LinearRegression().fit(X, y)
            slopes_lstm_logit.append(reg.coef_[0])
        
        # Logit → Weight
        X = data.policy_logit[:, i].reshape(-1, 1)
        y = data.portfolio_weight[:, i]
        if np.std(X) > 1e-6 and np.std(y) > 1e-6:
            reg = LinearRegression().fit(X, y)
            slopes_logit_weight.append(reg.coef_[0])
    
    results = {
        "TS_to_LSTM": {
            "mean_slope": float(np.mean(slopes_ts_lstm)) if slopes_ts_lstm else 0.0,
            "std_slope": float(np.std(slopes_ts_lstm)) if slopes_ts_lstm else 0.0,
            "pct_positive": float(100 * np.mean([s > 0 for s in slopes_ts_lstm])) if slopes_ts_lstm else 0.0,
        },
        "LSTM_to_Logit": {
            "mean_slope": float(np.mean(slopes_lstm_logit)) if slopes_lstm_logit else 0.0,
            "std_slope": float(np.std(slopes_lstm_logit)) if slopes_lstm_logit else 0.0,
            "pct_positive": float(100 * np.mean([s > 0 for s in slopes_lstm_logit])) if slopes_lstm_logit else 0.0,
        },
        "Logit_to_Weight": {
            "mean_slope": float(np.mean(slopes_logit_weight)) if slopes_logit_weight else 0.0,
            "std_slope": float(np.std(slopes_logit_weight)) if slopes_logit_weight else 0.0,
            "pct_positive": float(100 * np.mean([s > 0 for s in slopes_logit_weight])) if slopes_logit_weight else 0.0,
        },
    }
    
    print("\nPer-Stock Slope Distribution:")
    for key, vals in results.items():
        print(f"  {key}:")
        print(f"    Mean slope: {vals['mean_slope']:.4f} ± {vals['std_slope']:.4f}")
        print(f"    % Positive: {vals['pct_positive']:.1f}%")
    
    return results


def create_visualizations(data: TrendData, analyses: Dict, output_dir: Path):
    """
    Create comprehensive visualization suite.
    """
    print("\n=== Generating Visualizations ===")
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError as e:
        print(f"Matplotlib/seaborn not available: {e}. Skipping plots.")
        return
    
    # 1. Correlation Pyramid Diagram
    fig, ax = plt.subplots(figsize=(8, 10))
    
    pyramid_data = analyses["correlation_pyramid"]
    levels = [
        ("TS-Momentum", ""),
        ("", f"ρ={pyramid_data['TS_to_LSTM']['spearman']:.3f}"),
        ("LSTM-Trend", ""),
        ("", f"ρ={pyramid_data['LSTM_to_Logit']['spearman']:.3f}"),
        ("Policy-Logit", ""),
        ("", f"ρ={pyramid_data['Logit_to_Weight']['spearman']:.3f}"),
        ("Portfolio-Weight", ""),
    ]
    
    for i, (label, corr) in enumerate(levels):
        y = 1 - i * 0.15
        if label:
            ax.text(0.5, y, label, ha='center', va='center', fontsize=14, 
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        elif corr:
            ax.annotate('', xy=(0.5, y+0.05), xytext=(0.5, y+0.10),
                       arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
            ax.text(0.7, y+0.075, corr, fontsize=11, color='darkred', weight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.2)
    ax.axis('off')
    ax.set_title('Three-Layer Causal Chain: Trend → Weight\n(Spearman Correlations)', 
                fontsize=16, weight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / "trend_correlation_pyramid.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: trend_correlation_pyramid.png")
    
    # 2. Scatter Plots (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Flatten data for plotting
    ts_flat = data.ts_momentum.flatten()
    lstm_flat = data.lstm_trend.flatten()
    logit_flat = data.policy_logit.flatten()
    weight_flat = data.portfolio_weight.flatten()
    
    # Sample for plotting (if too many points)
    n_samples = min(10000, len(ts_flat))
    idx = np.random.choice(len(ts_flat), n_samples, replace=False)
    
    # Plot 1: TS vs LSTM
    axes[0, 0].scatter(ts_flat[idx], lstm_flat[idx], alpha=0.3, s=10)
    axes[0, 0].set_xlabel('TS-Momentum', fontsize=12)
    axes[0, 0].set_ylabel('LSTM-Trend', fontsize=12)
    axes[0, 0].set_title(f"TS → LSTM (ρ={pyramid_data['TS_to_LSTM']['spearman']:.3f})", fontsize=13)
    z = np.polyfit(ts_flat[idx], lstm_flat[idx], 1)
    p = np.poly1d(z)
    x_line = np.linspace(ts_flat[idx].min(), ts_flat[idx].max(), 100)
    axes[0, 0].plot(x_line, p(x_line), 'r--', lw=2, alpha=0.8)
    
    # Plot 2: LSTM vs Logit
    axes[0, 1].scatter(lstm_flat[idx], logit_flat[idx], alpha=0.3, s=10, color='green')
    axes[0, 1].set_xlabel('LSTM-Trend', fontsize=12)
    axes[0, 1].set_ylabel('Policy-Logit', fontsize=12)
    axes[0, 1].set_title(f"LSTM → Logit (ρ={pyramid_data['LSTM_to_Logit']['spearman']:.3f})", fontsize=13)
    z = np.polyfit(lstm_flat[idx], logit_flat[idx], 1)
    p = np.poly1d(z)
    x_line = np.linspace(lstm_flat[idx].min(), lstm_flat[idx].max(), 100)
    axes[0, 1].plot(x_line, p(x_line), 'r--', lw=2, alpha=0.8)
    
    # Plot 3: Logit vs Weight
    axes[1, 0].scatter(logit_flat[idx], weight_flat[idx], alpha=0.3, s=10, color='purple')
    axes[1, 0].set_xlabel('Policy-Logit', fontsize=12)
    axes[1, 0].set_ylabel('Portfolio-Weight', fontsize=12)
    axes[1, 0].set_title(f"Logit → Weight (ρ={pyramid_data['Logit_to_Weight']['spearman']:.3f})", fontsize=13)
    z = np.polyfit(logit_flat[idx], weight_flat[idx], 1)
    p = np.poly1d(z)
    x_line = np.linspace(logit_flat[idx].min(), logit_flat[idx].max(), 100)
    axes[1, 0].plot(x_line, p(x_line), 'r--', lw=2, alpha=0.8)
    
    # Plot 4: TS vs Weight (direct)
    axes[1, 1].scatter(ts_flat[idx], weight_flat[idx], alpha=0.3, s=10, color='orange')
    axes[1, 1].set_xlabel('TS-Momentum', fontsize=12)
    axes[1, 1].set_ylabel('Portfolio-Weight', fontsize=12)
    axes[1, 1].set_title(f"TS → Weight (Direct) (ρ={pyramid_data['TS_to_Weight']['spearman']:.3f})", fontsize=13)
    z = np.polyfit(ts_flat[idx], weight_flat[idx], 1)
    p = np.poly1d(z)
    x_line = np.linspace(ts_flat[idx].min(), ts_flat[idx].max(), 100)
    axes[1, 1].plot(x_line, p(x_line), 'r--', lw=2, alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(output_dir / "trend_scatter_plots.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: trend_scatter_plots.png")
    
    # 3. Quantile Uplift Bar Charts (3 panels)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    quantile_data = analyses["quantile_uplift"]
    
    for idx, (layer_name, ax) in enumerate(zip(
        ["TS_Momentum", "LSTM_Trend", "Policy_Logit"],
        axes
    )):
        layer_results = quantile_data[layer_name]
        bars = ax.bar(range(len(layer_results["mean_weights"])), 
                      layer_results["mean_weights"],
                      color=plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(layer_results["mean_weights"]))))
        ax.set_xlabel('Quantile', fontsize=12)
        ax.set_ylabel('Mean Portfolio Weight', fontsize=12)
        ax.set_title(f"{layer_name.replace('_', '-')}\nUplift: {layer_results['uplift_ratio']:.2f}x", 
                    fontsize=13, weight='bold')
        ax.set_xticks(range(len(layer_results["quantile_labels"])))
        ax.set_xticklabels(layer_results["quantile_labels"], rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "trend_quantile_uplift.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: trend_quantile_uplift.png")
    
    # 4. Time-Series Example (select 3 stocks: high/medium/low avg trend)
    T, N = data.ts_momentum.shape
    avg_ts_momentum = data.ts_momentum.mean(axis=0)
    
    high_idx = int(np.argmax(avg_ts_momentum))
    low_idx = int(np.argmin(avg_ts_momentum))
    mid_idx = int(np.argsort(avg_ts_momentum)[N // 2])
    
    fig, axes = plt.subplots(5, 3, figsize=(18, 15))
    fig.suptitle('Time-Series View: Three Representative Stocks', fontsize=16, weight='bold')
    
    for col_idx, stock_idx in enumerate([high_idx, mid_idx, low_idx]):
        stock_name = data.stock_names[stock_idx]
        avg_mom = avg_ts_momentum[stock_idx]
        
        axes[0, col_idx].plot(data.decision_steps, data.ts_momentum[:, stock_idx], 'o-', lw=2)
        axes[0, col_idx].set_title(f"{stock_name}\nAvg TS-Mom: {avg_mom:.4f}", fontsize=11, weight='bold')
        axes[0, col_idx].set_ylabel('TS-Momentum', fontsize=10)
        axes[0, col_idx].grid(alpha=0.3)
        
        axes[1, col_idx].plot(data.decision_steps, data.lstm_trend[:, stock_idx], 'o-', color='green', lw=2)
        axes[1, col_idx].set_ylabel('LSTM-Trend', fontsize=10)
        axes[1, col_idx].grid(alpha=0.3)
        
        axes[2, col_idx].plot(data.decision_steps, data.policy_logit[:, stock_idx], 'o-', color='purple', lw=2)
        axes[2, col_idx].set_ylabel('Policy-Logit', fontsize=10)
        axes[2, col_idx].grid(alpha=0.3)
        
        axes[3, col_idx].plot(data.decision_steps, data.portfolio_weight[:, stock_idx], 'o-', color='red', lw=2)
        axes[3, col_idx].set_ylabel('Portfolio-Weight', fontsize=10)
        axes[3, col_idx].grid(alpha=0.3)
        
        # Overlay all four on bottom panel for comparison
        ax_all = axes[4, col_idx]
        ax_all.plot(data.decision_steps, 
                   (data.ts_momentum[:, stock_idx] - data.ts_momentum[:, stock_idx].mean()) / (data.ts_momentum[:, stock_idx].std() + 1e-6),
                   'o-', label='TS-Mom (z)', alpha=0.7)
        ax_all.plot(data.decision_steps,
                   (data.lstm_trend[:, stock_idx] - data.lstm_trend[:, stock_idx].mean()) / (data.lstm_trend[:, stock_idx].std() + 1e-6),
                   's-', label='LSTM (z)', alpha=0.7)
        ax_all.plot(data.decision_steps,
                   (data.policy_logit[:, stock_idx] - data.policy_logit[:, stock_idx].mean()) / (data.policy_logit[:, stock_idx].std() + 1e-6),
                   '^-', label='Logit (z)', alpha=0.7)
        ax_all.plot(data.decision_steps,
                   (data.portfolio_weight[:, stock_idx] - data.portfolio_weight[:, stock_idx].mean()) / (data.portfolio_weight[:, stock_idx].std() + 1e-6),
                   'd-', label='Weight (z)', alpha=0.7, lw=2)
        ax_all.set_ylabel('Standardized', fontsize=10)
        ax_all.set_xlabel('Decision Step', fontsize=10)
        ax_all.legend(fontsize=8, loc='best')
        ax_all.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "trend_time_series_examples.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: trend_time_series_examples.png")
    
    # 5. Heatmap: Average trend vs average weight per stock
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Compute averages per stock
    avg_ts = data.ts_momentum.mean(axis=0)
    avg_lstm = data.lstm_trend.mean(axis=0)
    avg_logit = data.policy_logit.mean(axis=0)
    avg_weight = data.portfolio_weight.mean(axis=0)
    
    # Sort by TS momentum
    sort_idx = np.argsort(avg_ts)
    
    for idx, (vals, title) in enumerate([
        (avg_ts[sort_idx], "TS-Momentum"),
        (avg_lstm[sort_idx], "LSTM-Trend"),
        (avg_logit[sort_idx], "Policy-Logit"),
        (avg_weight[sort_idx], "Portfolio-Weight"),
    ]):
        im = axes[idx].imshow(vals.reshape(-1, 1), aspect='auto', cmap='RdYlGn', 
                             vmin=np.percentile(vals, 5), vmax=np.percentile(vals, 95))
        axes[idx].set_title(title, fontsize=13, weight='bold')
        axes[idx].set_xlabel('Avg Value', fontsize=11)
        axes[idx].set_ylabel('Stock (sorted by TS-Mom)', fontsize=11)
        axes[idx].set_yticks([])
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(output_dir / "trend_heatmap_per_stock.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: trend_heatmap_per_stock.png")
    
    print("  ✓ All visualizations complete")


def save_results(data: TrendData, analyses: Dict, output_dir: Path, args: argparse.Namespace):
    """Save comprehensive results to JSON."""
    print("\n=== Saving Results ===")
    
    results = {
        "metadata": {
            "model_path": args.model_path,
            "test_period": f"{args.test_start_date} to {args.test_end_date}",
            "num_stocks": len(data.stock_names),
            "num_decisions": len(data.decision_steps),
            "stock_names": data.stock_names,
        },
        "analyses": analyses,
        "summary": {
            "conclusion": "Three-layer causal chain validated" if all([
                analyses["correlation_pyramid"]["TS_to_LSTM"]["spearman"] > 0.3,
                analyses["correlation_pyramid"]["LSTM_to_Logit"]["spearman"] > 0.3,
                analyses["correlation_pyramid"]["Logit_to_Weight"]["spearman"] > 0.3,
            ]) else "Weak causal chain - investigate further",
            "key_findings": [
                f"TS-Momentum correlates with LSTM-Trend (ρ={analyses['correlation_pyramid']['TS_to_LSTM']['spearman']:.3f})",
                f"LSTM-Trend drives Policy-Logit (ρ={analyses['correlation_pyramid']['LSTM_to_Logit']['spearman']:.3f})",
                f"Policy-Logit determines Weight (ρ={analyses['correlation_pyramid']['Logit_to_Weight']['spearman']:.3f})",
                f"Direct TS→Weight correlation: ρ={analyses['correlation_pyramid']['TS_to_Weight']['spearman']:.3f}",
                f"Top-quantile TS-Momentum receives {analyses['quantile_uplift']['TS_Momentum']['uplift_ratio']:.2f}x more weight than bottom",
            ],
        },
    }
    
    output_path = output_dir / "trend_explainability_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"  Saved: trend_explainability_results.json")
    print("\n✓ Analysis complete!")
    print(f"\nKey Finding: {'Strong' if 'validated' in results['summary']['conclusion'] else 'Weak'} causal chain from TS-Momentum → Portfolio-Weight")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Three-Layer Trend Explainability: TS-Momentum → LSTM-Trend → Policy-Logit → Weight"
    )
    parser.add_argument("--model-path", required=True, help="Path to trained PPO .zip file")
    parser.add_argument("--market", default="hs300", help="Market code")
    parser.add_argument("--horizon", default="1", help="Prediction horizon")
    parser.add_argument("--relation-type", default="hy", help="Relation type")
    parser.add_argument("--test-start-date", required=True, help="Test start (YYYY-MM-DD)")
    parser.add_argument("--test-end-date", required=True, help="Test end (YYYY-MM-DD)")
    parser.add_argument("--data-root", default="dataset_default", help="Dataset root dir")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy")
    parser.add_argument("--max-steps", type=int, default=None, help="Max rollout steps")
    parser.add_argument("--ind-yn", action="store_true", default=True, help="Enable industry")
    parser.add_argument("--pos-yn", action="store_true", default=True, help="Enable positive")
    parser.add_argument("--neg-yn", action="store_true", default=True, help="Enable negative")
    parser.add_argument("--output-dir", default="./explainability_results", help="Output directory")
    parser.add_argument("--tickers-csv", default="tickers.csv", help="Ticker CSV file")
    parser.add_argument("--n-quantiles", type=int, default=10, help="Number of quantiles for uplift analysis")
    
    return parser.parse_args(argv)


def load_tickers(csv_path: str, num_stocks: int) -> List[str]:
    """Load ticker names from CSV."""
    import csv
    
    candidates = [Path(csv_path).expanduser()]
    if not candidates[0].is_absolute():
        candidates.append((REPO_ROOT / csv_path).expanduser())
    
    for candidate in candidates:
        if candidate.exists():
            try:
                with candidate.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    if "ticker" in (reader.fieldnames or []):
                        tickers = [row["ticker"].strip() for row in reader if row.get("ticker", "").strip()]
                        if len(tickers) >= num_stocks:
                            return tickers[:num_stocks]
            except Exception as e:
                print(f"Warning: Could not load tickers from {candidate}: {e}")
    
    # Fallback to generic names
    return [f"Stock_{i}" for i in range(num_stocks)]


def main(argv=None):
    args = parse_args(argv)
    
    # Auto-detect metadata (same as attention_viz)
    base_dir = Path(args.data_root).expanduser().resolve()
    data_dir = base_dir / f"data_train_predict_{args.market}" / f"{args.horizon}_{args.relation_type}"
    
    print("=== Three-Layer Trend Explainability ===")
    print(f"Model: {args.model_path}")
    print(f"Dataset: {data_dir}")
    print(f"Test period: {args.test_start_date} to {args.test_end_date}")
    
    # Load dataset
    test_dataset = AllGraphDataSampler(
        base_dir=str(data_dir),
        date=True,
        test_start_date=args.test_start_date,
        test_end_date=args.test_end_date,
        mode="test",
    )
    
    if len(test_dataset) == 0:
        raise RuntimeError("Empty test dataset")
    
    # Infer num_stocks from first sample
    import pickle
    sample_files = sorted(data_dir.glob("*.pkl"))
    with sample_files[0].open("rb") as f:
        sample = pickle.load(f)
        num_stocks = sample["features"].shape[-2]
        input_dim = sample["features"].shape[-1]
    
    args.num_stocks = num_stocks
    args.input_dim = input_dim
    
    print(f"Num stocks: {num_stocks}")
    print(f"Feature dim: {input_dim}")
    
    # Load tickers
    stock_names = load_tickers(args.tickers_csv, num_stocks)
    
    # Load model
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), pin_memory=True)
    model = PPO.load(args.model_path, env=None, device=args.device)
    
    # Collect trend data
    data = collect_trend_data(test_loader, model, args, torch.device(args.device), stock_names)
    
    # Run analyses
    analyses = {
        "correlation_pyramid": analyze_correlation_pyramid(data),
        "quantile_uplift": analyze_quantile_uplift(data, n_quantiles=args.n_quantiles),
        "path_regressions": analyze_path_regressions(data),
        "per_stock_slopes": analyze_per_stock_slopes(data),
    }
    
    # Create output directory
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    create_visualizations(data, analyses, output_dir)
    
    # Save results
    save_results(data, analyses, output_dir, args)


if __name__ == "__main__":
    main()
