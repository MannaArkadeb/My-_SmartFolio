# Persistency Integration for SmartFolio

## Overview

SmartFolio now supports **Persistency** - a fine-tuning mechanism that prevents catastrophic forgetting during incremental learning. When fine-tuning a baseline model on new data, the Persistency feature adds an L2 regularization penalty to keep the fine-tuned parameters close to the baseline model's weights.

This is particularly useful for:
- **Monthly fine-tuning**: Update models incrementally as new data arrives
- **Continual learning**: Adapt to new market conditions without losing learned strategies
- **Risk management**: Prevent overfitting to recent data while preserving long-term knowledge

## Implementation Details

### Core Component: PersistentPPO

The `trainer/persistent_ppo.py` module implements `PersistentPPO`, a custom PPO variant that extends `stable_baselines3.PPO`:

```python
from trainer.persistent_ppo import PersistentPPO

model = PersistentPPO(
    policy='MlpPolicy',
    env=env,
    baseline_checkpoint='checkpoints/baseline.zip',  # Path to baseline model
    persistency_lambda=1e-4,                         # Regularization strength
    **standard_ppo_params
)
```

**Key Features:**
- Loads baseline model parameters at initialization
- Computes L2 penalty: `penalty = ||theta - theta_baseline||²`
- Adds penalty to training loss: `total_loss = ppo_loss + lambda * penalty`
- Logs persistency metrics to tensorboard

### Loss Function

The total training loss becomes:

```
L_total = L_ppo + λ * Σᵢ ||θᵢ - θᵢ_baseline||²

where:
- L_ppo: Standard PPO loss (policy + value + entropy)
- λ: persistency_lambda (controls regularization strength)
- θᵢ: Current policy parameters
- θᵢ_baseline: Baseline policy parameters
```

## Usage Guide

### 1. Basic Fine-Tuning with Persistency

Train a new model or fine-tune an existing one with persistency enabled:

```bash
python main.py \
    --policy MLP \
    --baseline_checkpoint ./checkpoints/baseline.zip \
    --persistency_lambda 1e-4 \
    --resume_model_path ./checkpoints/previous_model.zip \
    --save_dir ./checkpoints
```

**Parameters:**
- `--baseline_checkpoint`: Path to the baseline model (.zip file from SB3)
- `--persistency_lambda`: Regularization coefficient (recommended: 1e-5 to 1e-3)
  - `0.0`: Disabled (default, standard PPO training)
  - `1e-5`: Very weak regularization (maximum adaptation)
  - `1e-4`: Moderate regularization (balanced approach)
  - `1e-3`: Strong regularization (minimal deviation from baseline)
- `--resume_model_path`: Optional starting checkpoint to continue training from

### 2. Monthly Incremental Fine-Tuning

Fine-tune models month-by-month with persistency to maintain stability:

```bash
python main.py \
    --run_monthly_fine_tune \
    --baseline_checkpoint ./checkpoints/baseline.zip \
    --persistency_lambda 5e-4 \
    --fine_tune_steps 5000 \
    --promotion_min_sharpe 0.5 \
    --promotion_max_drawdown 0.2
```

**Workflow:**
1. Loads baseline checkpoint for persistency
2. Fine-tunes on each monthly batch
3. Evaluates performance metrics (Sharpe ratio, drawdown)
4. Promotes checkpoint if it meets gating criteria
5. Uses promoted checkpoint as baseline for next month

### 3. Training from Scratch (No Persistency)

Standard PPO training without persistency:

```bash
python main.py \
    --policy HGAT \
    --persistency_lambda 0.0
    # or simply omit --baseline_checkpoint
```

### 4. HGAT Policy with Persistency

Use custom policy architecture with persistency:

```bash
python main.py \
    --policy HGAT \
    --ind_yn y --pos_yn y --neg_yn y \
    --baseline_checkpoint ./checkpoints/hgat_baseline.zip \
    --persistency_lambda 2e-4 \
    --num_stocks 50
```

## Configuration Examples

### Conservative Fine-Tuning
Minimal deviation from baseline (for stable markets):
```bash
--persistency_lambda 1e-3
--promotion_min_sharpe 0.6
--promotion_max_drawdown 0.15
```

### Balanced Fine-Tuning
Standard configuration (recommended starting point):
```bash
--persistency_lambda 1e-4
--promotion_min_sharpe 0.5
--promotion_max_drawdown 0.2
```

### Aggressive Adaptation
Maximum adaptation to new data (for rapidly changing markets):
```bash
--persistency_lambda 1e-5
--promotion_min_sharpe 0.4
--promotion_max_drawdown 0.25
```

## Integration Points

### Files Modified

1. **`trainer/persistent_ppo.py`** (NEW)
   - Core implementation of PersistentPPO
   - Baseline parameter loading and penalty computation
   - Custom training loop with L2 regularization

2. **`trainer/trainer.py`**
   - Import PersistentPPO
   - Modified `train_model_one()` to use PersistentPPO when configured
   - Conditional model creation based on persistency parameters

3. **`trainer/irl_trainer.py`**
   - Import PersistentPPO for IRL-based training
   - Compatible with inverse RL reward networks

4. **`main.py`**
   - Added CLI arguments: `--persistency_lambda`
   - Updated model creation logic in `train_predict()`
   - Support for both MLP and HGAT policies

### Promotion Gate Integration

Persistency works seamlessly with the existing promotion gate mechanism:

```python
from trainer.evaluation_utils import apply_promotion_gate

# After fine-tuning with persistency
final_eval = model_predict(args, model, test_loader, split="final_test")

# Promote if metrics meet criteria
apply_promotion_gate(
    args,
    candidate_path='./checkpoints/finetuned.zip',
    summary_metrics=final_eval.get("summary"),
    log_info=final_eval.get("log")
)
```

## Monitoring and Logging

### TensorBoard Metrics

When training with persistency enabled, additional metrics are logged:

```
train/persistency_loss      # L2 penalty value
train/persistency_lambda    # Lambda coefficient
train/policy_gradient_loss  # Standard PPO policy loss
train/value_loss           # Value function loss
train/entropy_loss         # Entropy bonus
```

View in TensorBoard:
```bash
tensorboard --logdir ./logs
```

### Console Output

During training, you'll see:
```
[PersistentPPO] Loaded baseline from checkpoints/baseline.zip (234 parameters)
[PersistentPPO] Persistency lambda: 1.00e-04
```

## Advanced Usage

### Programmatic API

Use PersistentPPO directly in custom training scripts:

```python
from trainer.persistent_ppo import PersistentPPO

# Create model with persistency
model = PersistentPPO(
    policy='MlpPolicy',
    env=env,
    baseline_checkpoint='path/to/baseline.zip',
    persistency_lambda=1e-4,
    persistency_device='cpu',  # Store baseline on CPU to save GPU memory
    n_steps=2048,
    batch_size=64,
    learning_rate=3e-4,
    verbose=1
)

# Train with persistency penalty
model.learn(total_timesteps=100000)

# Save fine-tuned model
model.save('path/to/finetuned.zip')

# Load and continue training
model = PersistentPPO.load(
    'path/to/finetuned.zip',
    env=env,
    baseline_checkpoint='path/to/baseline.zip',
    persistency_lambda=1e-4
)
```

### Custom Baseline Selection

You can dynamically select baseline checkpoints:

```python
import os
from pathlib import Path

# Use most recent checkpoint as baseline
checkpoint_dir = Path('./checkpoints')
checkpoints = sorted(checkpoint_dir.glob('model_*.zip'))
baseline = str(checkpoints[-1]) if checkpoints else None

if baseline:
    model = PersistentPPO(
        policy='MlpPolicy',
        env=env,
        baseline_checkpoint=baseline,
        persistency_lambda=1e-4
    )
```

### Hyperparameter Tuning

Recommended ranges for `persistency_lambda`:

| Lambda Value | Behavior | Use Case |
|-------------|----------|----------|
| 0.0 | No regularization | Initial training, major distribution shift |
| 1e-5 | Very weak | Rapid market changes, high adaptability needed |
| 1e-4 | **Moderate (recommended)** | Standard monthly fine-tuning |
| 5e-4 | Strong | Conservative updates, stable markets |
| 1e-3+ | Very strong | Minimal updates, preserve baseline behavior |

## Troubleshooting

### Issue: Baseline checkpoint not found
```
Warning: Baseline checkpoint not found: <path>. Persistency penalty will be disabled.
```
**Solution:** Ensure the baseline checkpoint path is correct and the file exists.

### Issue: Shape mismatch
```
Shape mismatch for parameter 'policy.mlp_extractor.policy_net.0.weight': 
current torch.Size([64, 128]) vs baseline torch.Size([32, 128])
```
**Solution:** The baseline and current model architectures must match. Ensure:
- Same policy type (MLP/HGAT)
- Same number of stocks
- Same hidden dimensions
- Same policy_kwargs

### Issue: No persistency penalty applied
```
No matching parameters found between current policy and baseline.
```
**Solution:** Check that:
- `persistency_lambda > 0`
- Baseline checkpoint loads successfully
- Policy architecture matches baseline

### Issue: Training is too conservative
Model doesn't adapt to new data.

**Solution:** Reduce `persistency_lambda` (try 1e-5 or 5e-5).

### Issue: Training is unstable
Model diverges from baseline too quickly.

**Solution:** Increase `persistency_lambda` (try 5e-4 or 1e-3).

## Performance Considerations

### Memory Usage
- Baseline parameters are stored on CPU by default to save GPU memory
- Can be changed via `persistency_device='cuda:0'` for faster penalty computation
- Adds ~50-100MB overhead for typical models

### Training Speed
- Persistency penalty adds ~5-10% overhead to training time
- Penalty computation is highly optimized (single pass over parameters)
- Negligible impact compared to environment simulation

### Best Practices
1. Start with `persistency_lambda=1e-4` and adjust based on results
2. Use validation set to tune lambda (balance stability vs. adaptation)
3. Store baselines on CPU unless GPU memory is abundant
4. Monitor `train/persistency_loss` to ensure penalty is active
5. Compare fine-tuned models with and without persistency

## Related Configuration

### Promotion Gate Settings
Control which checkpoints become the new baseline:

```bash
--promotion_min_sharpe 0.5        # Minimum Sharpe ratio
--promotion_max_drawdown 0.2      # Maximum 20% drawdown
--baseline_checkpoint ./checkpoints/baseline.zip
```

### Risk Profile Settings
Persistency works with risk-adaptive policies:

```bash
--risk_score 0.5                  # Moderate risk (0=conservative, 1=aggressive)
--dd_base_weight 1.0              # Drawdown penalty weight
--dd_risk_factor 1.0              # Risk adaptation factor
```

## Examples

### Example 1: Initial Training (No Persistency)
```bash
python main.py \
    --policy HGAT \
    --market CSI300 \
    --train_start_date 2018-01-01 \
    --train_end_date 2022-12-31 \
    --save_dir ./checkpoints
```

### Example 2: Monthly Fine-Tuning with Persistency
```bash
python main.py \
    --run_monthly_fine_tune \
    --policy HGAT \
    --baseline_checkpoint ./checkpoints/baseline.zip \
    --persistency_lambda 1e-4 \
    --fine_tune_steps 5000 \
    --market CSI300 \
    --test_start_date 2023-01-01 \
    --test_end_date 2023-12-31
```

### Example 3: Resume Training with Persistency
```bash
python main.py \
    --policy MLP \
    --resume_model_path ./checkpoints/model_2023-06.zip \
    --baseline_checkpoint ./checkpoints/baseline.zip \
    --persistency_lambda 2e-4 \
    --train_start_date 2023-07-01 \
    --train_end_date 2023-07-31 \
    --save_dir ./checkpoints
```

## Technical Details

### Parameter Matching
PersistentPPO matches parameters by name between current policy and baseline:
- Only matching parameters contribute to penalty
- Unmatched parameters are skipped (with warning if verbose > 0)
- Shape compatibility is verified before computing penalty

### Gradient Flow
The persistency penalty affects gradients during backpropagation:
```python
# Standard PPO loss
loss_ppo = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

# Add persistency penalty
penalty = sum(||param - baseline_param||² for param in policy.parameters())
loss_total = loss_ppo + persistency_lambda * penalty

# Backprop through total loss
loss_total.backward()
```

### Checkpoint Compatibility
- Baseline must be a valid SB3 checkpoint (.zip file)
- Compatible with all SB3-supported policies
- Custom policies (HGAT) must match architecture exactly

## References

- **Pathway Persistence**: Inspired by Pathway's state persistence mechanisms for streaming computation
- **Elastic Weight Consolidation (EWC)**: Similar concept for continual learning in neural networks
- **Fine-tuning with Regularization**: Standard technique in transfer learning

## Support

For issues or questions:
1. Check console output for warnings/errors
2. Verify baseline checkpoint compatibility
3. Tune `persistency_lambda` based on validation metrics
4. Review TensorBoard logs for persistency_loss trends

---

**Version:** 1.0  
**Last Updated:** November 28, 2025  
**Author:** SmartFolio Team
