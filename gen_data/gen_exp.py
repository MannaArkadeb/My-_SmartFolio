"""
Integration Example: Using Hybrid Ensemble Expert Strategy

This file demonstrates how to integrate the new hybrid ensemble expert strategy
with your existing codebase.
"""

import numpy as np
import pandas as pd
import pickle
import os
from expert_strategy_ensemble import (
    HybridEnsembleExpert,
    generate_expert_trajectories,
    save_expert_trajectories,
    load_expert_trajectories
)


# Example 1: Generate expert trajectories for training
def example_generate_trajectories_for_dataset():
    """
    Example showing how to generate expert trajectories using your dataset
    """
    
    class Args:
        market = 'custom'
        input_dim = 6
        ind_yn = True
        pos_yn = True
        neg_yn = True
    
    args = Args()
    
    # Import your data loader (this would be your actual import)
    # from dataloader.data_loader import AllGraphDataSampler
    
    # For demonstration, we'll create a mock dataset class
    class MockDataset:
        def _init_(self, n_samples=100):
            self.n_samples = n_samples
        
        def _len_(self):
            return self.n_samples
        
        def _getitem_(self, idx):
            n_stocks = 50  # Example: 50 stocks
            n_features = 6
            
            # Mock data structure matching your actual data
            return {
                'features': type('obj', (object,), {
                    'numpy': lambda: np.random.randn(n_stocks, n_features)
                }),
                'labels': type('obj', (object,), {
                    'numpy': lambda: np.random.randn(n_stocks) * 0.01
                }),
                'corr': type('obj', (object,), {
                    'numpy': lambda: self._create_corr_matrix(n_stocks)
                }),
                'industry_matrix': type('obj', (object,), {
                    'numpy': lambda: self._create_industry_matrix(n_stocks)
                }),
                'pos_matrix': type('obj', (object,), {
                    'numpy': lambda: self._create_sparse_matrix(n_stocks)
                }),
                'neg_matrix': type('obj', (object,), {
                    'numpy': lambda: self._create_sparse_matrix(n_stocks)
                }),
            }
        
        def _create_corr_matrix(self, n):
            corr = np.random.rand(n, n)
            corr = (corr + corr.T) / 2
            np.fill_diagonal(corr, 1.0)
            return corr
        
        def _create_industry_matrix(self, n):
            ind = np.random.rand(n, n)
            ind = (ind > 0.7).astype(float)
            return ind
        
        def _create_sparse_matrix(self, n):
            mat = np.random.rand(n, n)
            mat = (mat > 0.8).astype(float)
            return mat
    
    # Create mock dataset
    train_dataset = MockDataset(n_samples=100)
    
    print("Generating expert trajectories...")
    
    # Generate expert trajectories
    expert_trajectories = generate_expert_trajectories(
        args, 
        train_dataset, 
        num_trajectories=50  # Start with fewer trajectories for testing
    )
    
    print(f"Generated {len(expert_trajectories)} expert trajectories")
    
    # Save trajectories
    save_path = f'expert_trajectories_{args.market}.pkl'
    save_expert_trajectories(expert_trajectories, save_path)
    print(f"Expert trajectories saved to {save_path}")
    
    # Example: Load and inspect
    loaded_trajectories = load_expert_trajectories(save_path)
    print(f"\nLoaded {len(loaded_trajectories)} trajectories")
    
    # Inspect first trajectory
    state, action = loaded_trajectories[0]
    print(f"State shape: {state.shape}")
    print(f"Action shape: {action.shape}")
    print(f"Number of selected stocks: {action.sum()}")
    
    return expert_trajectories


# Example 2: Customize ensemble weights
def example_custom_ensemble_weights():
    """
    Example showing how to customize ensemble weights for different strategies
    """
    
    # Strategy 1: Conservative (more weight on Risk Parity and HRP)
    conservative_weights = {
        'robust_markowitz': 0.15,
        'black_litterman': 0.15,
        'risk_parity': 0.35,
        'hrp': 0.30,
        'multi_period': 0.05
    }
    
    # Strategy 2: Aggressive (more weight on Markowitz and Black-Litterman)
    aggressive_weights = {
        'robust_markowitz': 0.40,
        'black_litterman': 0.35,
        'risk_parity': 0.10,
        'hrp': 0.10,
        'multi_period': 0.05
    }
    
    # Strategy 3: Balanced
    balanced_weights = {
        'robust_markowitz': 0.30,
        'black_litterman': 0.25,
        'risk_parity': 0.20,
        'hrp': 0.15,
        'multi_period': 0.10
    }
    
    print("Testing different ensemble strategies...\n")
    
    # Create dummy data
    n_stocks = 50
    returns = np.random.randn(n_stocks) * 0.01
    correlation_matrix = np.random.rand(n_stocks, n_stocks)
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    np.fill_diagonal(correlation_matrix, 1.0)
    industry_matrix = (np.random.rand(n_stocks, n_stocks) > 0.7).astype(float)
    
    # Test each strategy
    for name, weights in [('Conservative', conservative_weights),
                          ('Aggressive', aggressive_weights),
                          ('Balanced', balanced_weights)]:
        
        expert = HybridEnsembleExpert(
            ensemble_weights=weights,
            randomize_params=False  # Fixed params for comparison
        )
        
        actions = expert.generate_expert_action(
            returns=returns,
            correlation_matrix=correlation_matrix,
            industry_matrix=industry_matrix
        )
        
        print(f"{name} Strategy:")
        print(f"  Selected stocks: {actions.sum()}")
        print(f"  Indices: {np.where(actions == 1)[0][:10]}...")
        print()


# Example 3: Generate diverse trajectories with parameter randomization
def example_diverse_trajectories():
    """
    Example showing how to generate diverse trajectories through parameter randomization
    """
    
    print("Generating diverse expert trajectories...\n")
    
    # Create expert with randomization enabled
    expert = HybridEnsembleExpert(randomize_params=True)
    
    # Generate multiple actions from the same state
    n_stocks = 50
    returns = np.random.randn(n_stocks) * 0.01
    correlation_matrix = np.random.rand(n_stocks, n_stocks)
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    np.fill_diagonal(correlation_matrix, 1.0)
    industry_matrix = (np.random.rand(n_stocks, n_stocks) > 0.7).astype(float)
    
    actions_list = []
    for i in range(5):
        actions = expert.generate_expert_action(
            returns=returns,
            correlation_matrix=correlation_matrix,
            industry_matrix=industry_matrix
        )
        actions_list.append(actions)
        print(f"Trajectory {i+1}: Selected {actions.sum()} stocks")
    
    # Calculate diversity (average Hamming distance)
    diversity_scores = []
    for i in range(len(actions_list)):
        for j in range(i+1, len(actions_list)):
            hamming_dist = np.sum(actions_list[i] != actions_list[j])
            diversity_scores.append(hamming_dist)
    
    avg_diversity = np.mean(diversity_scores)
    print(f"\nAverage diversity (Hamming distance): {avg_diversity:.2f}")
    print(f"Diversity as percentage: {avg_diversity / n_stocks * 100:.1f}%")


# Example 4: Integration with existing correlation calculation
def example_integration_with_correlation():
    """
    Example showing how to integrate with your existing correlation calculation
    """
    
    print("Integration with correlation calculation...\n")
    
    # Assume you have calculated correlations using your existing code
    # (from the second file you provided)
    
    # Mock correlation data (in practice, this would be loaded from your CSV files)
    market = 'custom'
    date = '2023-01-31'
    
    # In practice, you would load this from your correlation CSV:
    # corr_df = pd.read_csv(f"../dataset/corr/{market}/{date}.csv", index_col=0)
    
    # For demonstration, create mock data
    n_stocks = 50
    stock_codes = [f'stock_{i:03d}' for i in range(n_stocks)]
    corr_df = pd.DataFrame(
        np.random.rand(n_stocks, n_stocks),
        index=stock_codes,
        columns=stock_codes
    )
    corr_df = (corr_df + corr_df.T) / 2
    for i in range(n_stocks):
        corr_df.iloc[i, i] = 1.0
    
    # Load other required data
    returns = np.random.randn(n_stocks) * 0.01
    industry_matrix = (np.random.rand(n_stocks, n_stocks) > 0.7).astype(float)
    
    # Generate expert action
    expert = HybridEnsembleExpert()
    actions = expert.generate_expert_action(
        returns=returns,
        correlation_matrix=corr_df.values,
        industry_matrix=industry_matrix
    )
    
    # Get selected stock codes
    selected_stocks = [stock_codes[i] for i in range(n_stocks) if actions[i] == 1]
    
    print(f"Date: {date}")
    print(f"Selected {len(selected_stocks)} stocks:")
    print(f"Stocks: {selected_stocks[:10]}...")


# Example 5: Batch processing for multiple dates
def example_batch_processing():
    """
    Example showing how to process multiple dates in batch
    """
    
    print("Batch processing for multiple dates...\n")
    
    market = 'custom'
    dates = ['2023-01-31', '2023-02-28', '2023-03-31']
    
    all_trajectories = []
    
    for date in dates:
        print(f"Processing {date}...")
        
        # In practice, load your actual data here
        n_stocks = 50
        returns = np.random.randn(n_stocks) * 0.01
        correlation_matrix = np.random.rand(n_stocks, n_stocks)
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        np.fill_diagonal(correlation_matrix, 1.0)
        industry_matrix = (np.random.rand(n_stocks, n_stocks) > 0.7).astype(float)
        features = np.random.randn(n_stocks, 6)
        
        # Generate expert action
        expert = HybridEnsembleExpert(randomize_params=True)
        actions = expert.generate_expert_action(
            returns=returns,
            correlation_matrix=correlation_matrix,
            industry_matrix=industry_matrix
        )
        
        # Store as trajectory
        state = features
        all_trajectories.append({
            'date': date,
            'state': state,
            'action': actions
        })
        
        print(f"  Selected {actions.sum()} stocks\n")
    
    # Save all trajectories
    save_path = f'expert_trajectories_{market}_batch.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(all_trajectories, f)
    
    print(f"Saved {len(all_trajectories)} date-specific trajectories to {save_path}")


if _name_ == '_main_':
    print("="*60)
    print("Hybrid Ensemble Expert Strategy - Integration Examples")
    print("="*60)
    print()
    
    # Run examples
    print("\n" + "="*60)
    print("Example 1: Generate Expert Trajectories for Dataset")
    print("="*60)
    example_generate_trajectories_for_dataset()
    
    print("\n" + "="*60)
    print("Example 2: Custom Ensemble Weights")
    print("="*60)
    example_custom_ensemble_weights()
    
    print("\n" + "="*60)
    print("Example 3: Diverse Trajectories")
    print("="*60)
    example_diverse_trajectories()
    
    print("\n" + "="*60)
    print("Example 4: Integration with Correlation Calculation")
    print("="*60)
    example_integration_with_correlation()
    
    print("\n" + "="*60)
    print("Example 5: Batch Processing")
    print("="*60)
    example_batch_processing()
    
    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)
