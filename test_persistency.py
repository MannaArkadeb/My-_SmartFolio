"""
Quick test script to verify PersistentPPO integration.

This script performs a minimal smoke test to ensure:
1. PersistentPPO can be imported
2. Model creation works with and without persistency
3. Baseline loading functions correctly
4. Training loop executes without errors
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_import():
    """Test that PersistentPPO can be imported."""
    print("=" * 70)
    print("Test 1: Import PersistentPPO")
    print("=" * 70)
    
    try:
        from trainer.persistent_ppo import PersistentPPO
        print("✓ Successfully imported PersistentPPO")
        return True
    except Exception as e:
        print(f"✗ Failed to import: {e}")
        return False


def test_model_creation_no_persistency():
    """Test creating a standard PPO model without persistency."""
    print("\n" + "=" * 70)
    print("Test 2: Create Model Without Persistency")
    print("=" * 70)
    
    try:
        import gymnasium as gym
        from trainer.persistent_ppo import PersistentPPO
        
        # Create simple environment
        env = gym.make('CartPole-v1')
        
        # Create model without persistency (lambda=0)
        model = PersistentPPO(
            policy='MlpPolicy',
            env=env,
            persistency_lambda=0.0,
            verbose=1
        )
        
        print("✓ Created PersistentPPO model without persistency")
        print(f"  - Persistency enabled: {model._persistency_enabled}")
        print(f"  - Lambda: {model.persistency_lambda}")
        print(f"  - Baseline params loaded: {len(model.baseline_params)}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation_with_nonexistent_baseline():
    """Test creating a model with persistency but nonexistent baseline."""
    print("\n" + "=" * 70)
    print("Test 3: Create Model With Nonexistent Baseline")
    print("=" * 70)
    
    try:
        import gymnasium as gym
        from trainer.persistent_ppo import PersistentPPO
        
        env = gym.make('CartPole-v1')
        
        # Create model with persistency pointing to nonexistent baseline
        model = PersistentPPO(
            policy='MlpPolicy',
            env=env,
            baseline_checkpoint='./nonexistent_baseline.zip',
            persistency_lambda=1e-4,
            verbose=1
        )
        
        print("✓ Model created despite missing baseline (penalty disabled)")
        print(f"  - Persistency enabled: {model._persistency_enabled}")
        print(f"  - Lambda: {model.persistency_lambda}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_short_training_run():
    """Test a short training run without persistency."""
    print("\n" + "=" * 70)
    print("Test 4: Short Training Run (No Persistency)")
    print("=" * 70)
    
    try:
        import gymnasium as gym
        from trainer.persistent_ppo import PersistentPPO
        
        env = gym.make('CartPole-v1')
        
        model = PersistentPPO(
            policy='MlpPolicy',
            env=env,
            n_steps=128,
            batch_size=64,
            persistency_lambda=0.0,
            verbose=0
        )
        
        print("Training for 256 timesteps...")
        model.learn(total_timesteps=256)
        
        print("✓ Training completed successfully")
        
        # Test saving
        test_save_path = './test_checkpoint.zip'
        model.save(test_save_path)
        print(f"✓ Model saved to {test_save_path}")
        
        # Clean up
        if os.path.exists(test_save_path):
            os.remove(test_save_path)
            print("✓ Cleanup completed")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_persistency_with_real_baseline():
    """Test persistency with a real baseline checkpoint."""
    print("\n" + "=" * 70)
    print("Test 5: Training With Persistency (Real Baseline)")
    print("=" * 70)
    
    try:
        import gymnasium as gym
        from trainer.persistent_ppo import PersistentPPO
        from stable_baselines3 import PPO
        
        env = gym.make('CartPole-v1')
        
        # Step 1: Create and save a baseline model
        baseline_path = './test_baseline.zip'
        print("Creating baseline model...")
        baseline = PPO(
            policy='MlpPolicy',
            env=env,
            n_steps=128,
            batch_size=64,
            verbose=0
        )
        baseline.learn(total_timesteps=256)
        baseline.save(baseline_path)
        print(f"✓ Baseline saved to {baseline_path}")
        
        # Step 2: Create PersistentPPO with the baseline
        print("\nCreating PersistentPPO with baseline...")
        persistent_model = PersistentPPO(
            policy='MlpPolicy',
            env=env,
            baseline_checkpoint=baseline_path,
            persistency_lambda=1e-3,
            n_steps=128,
            batch_size=64,
            verbose=1
        )
        
        print(f"✓ PersistentPPO created")
        print(f"  - Persistency enabled: {persistent_model._persistency_enabled}")
        print(f"  - Baseline params: {len(persistent_model.baseline_params)}")
        
        # Step 3: Train with persistency
        print("\nTraining with persistency for 256 timesteps...")
        persistent_model.learn(total_timesteps=256)
        print("✓ Training with persistency completed")
        
        # Step 4: Verify persistency loss is computed
        penalty = persistent_model._compute_persistency_loss()
        print(f"✓ Persistency penalty computed: {penalty.item():.6f}")
        
        # Cleanup
        if os.path.exists(baseline_path):
            os.remove(baseline_path)
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Persistency test failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error
        baseline_path = './test_baseline.zip'
        if os.path.exists(baseline_path):
            os.remove(baseline_path)
        
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "PERSISTENCY INTEGRATION TEST SUITE" + " " * 19 + "║")
    print("╚" + "=" * 68 + "╝")
    
    tests = [
        ("Import Test", test_import),
        ("Model Creation (No Persistency)", test_model_creation_no_persistency),
        ("Model Creation (Missing Baseline)", test_model_creation_with_nonexistent_baseline),
        ("Short Training Run", test_short_training_run),
        ("Training with Persistency", test_persistency_with_real_baseline),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 27 + "TEST SUMMARY" + " " * 29 + "║")
    print("╚" + "=" * 68 + "╝")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} | {name}")
    
    print("-" * 70)
    print(f"TOTAL: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! Persistency integration is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed. Please review the output above.")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
