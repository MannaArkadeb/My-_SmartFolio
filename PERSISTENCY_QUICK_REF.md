# Persistency Quick Reference

## 🚀 Quick Start

### 1. Enable Persistency in Training
```bash
python main.py \
    --baseline_checkpoint ./checkpoints/baseline.zip \
    --persistency_lambda 1e-4
```

### 2. Run Monthly Fine-Tuning
```bash
python main.py \
    --run_monthly_fine_tune \
    --baseline_checkpoint ./checkpoints/baseline.zip \
    --persistency_lambda 1e-4 \
    --fine_tune_steps 5000
```

### 3. Test Implementation
```bash
python test_persistency.py
```

---

## 📋 Key Files

| File | Purpose |
|------|---------|
| `trainer/persistent_ppo.py` | Core implementation of PersistentPPO |
| `trainer/trainer.py` | Integration point (standard training) |
| `trainer/irl_trainer.py` | Integration point (IRL training) |
| `main.py` | CLI entry point with model creation |
| `PERSISTENCY_GUIDE.md` | Full documentation (500+ lines) |
| `test_persistency.py` | Automated test suite |

---

## ⚙️ Configuration

### CLI Arguments

```bash
--baseline_checkpoint PATH      # Path to baseline .zip checkpoint
--persistency_lambda FLOAT      # L2 penalty coefficient (default: 0.0)
```

### Recommended Lambda Values

| Value | Behavior | Use Case |
|-------|----------|----------|
| `0.0` | **Disabled** | Initial training, no baseline |
| `1e-5` | Very weak | Rapid market changes |
| `1e-4` | **Moderate (default)** | Standard fine-tuning |
| `5e-4` | Strong | Conservative updates |
| `1e-3` | Very strong | Minimal changes |

---

## 🔧 Programmatic API

```python
from trainer.persistent_ppo import PersistentPPO

# Create model with persistency
model = PersistentPPO(
    policy='MlpPolicy',
    env=env,
    baseline_checkpoint='path/to/baseline.zip',
    persistency_lambda=1e-4,
    persistency_device='cpu',  # Store baseline on CPU
    n_steps=2048,
    batch_size=64,
    verbose=1
)

# Train with persistency penalty
model.learn(total_timesteps=100000)

# Save fine-tuned model
model.save('path/to/finetuned.zip')
```

---

## 📊 Monitoring

### TensorBoard Metrics
```bash
tensorboard --logdir ./logs
```

**Key Metrics:**
- `train/persistency_loss` - L2 penalty value
- `train/persistency_lambda` - Lambda coefficient
- `train/policy_gradient_loss` - PPO policy loss
- `train/value_loss` - Value function loss

### Console Output
```
[PersistentPPO] Loaded baseline from checkpoints/baseline.zip (234 parameters)
[PersistentPPO] Persistency lambda: 1.00e-04
```

---

## 🏗️ Architecture

### Loss Function
```
L_total = L_ppo + λ * Σᵢ ||θᵢ - θᵢ_baseline||²
```

### Training Flow
```
Load baseline → Compute penalty → Add to loss → Backprop → Update weights
```

### Memory Usage
- Baseline on CPU: ~50-100MB overhead
- Minimal GPU impact (~5-10% slower training)

---

## ✅ Testing Checklist

- [ ] Run `python test_persistency.py` (all 5 tests pass)
- [ ] Train baseline model and save checkpoint
- [ ] Fine-tune with `--persistency_lambda 1e-4`
- [ ] Check TensorBoard for `train/persistency_loss`
- [ ] Verify fine-tuned model loads correctly
- [ ] Test monthly fine-tuning workflow

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Baseline not found | Check path, persistency auto-disabled |
| Shape mismatch | Ensure same architecture (num_stocks, hidden_dim) |
| No penalty applied | Verify lambda > 0 and baseline loads |
| Too conservative | Reduce lambda (try 1e-5) |
| Too unstable | Increase lambda (try 5e-4) |

---

## 📖 Documentation Links

- **Full Guide:** `PERSISTENCY_GUIDE.md` (500+ lines)
- **Implementation:** `IMPLEMENTATION_SUMMARY.md`
- **Code:** `trainer/persistent_ppo.py`
- **Tests:** `test_persistency.py`

---

## 🎯 Example Workflows

### Workflow 1: Initial Training (No Persistency)
```bash
python main.py \
    --policy HGAT \
    --market CSI300 \
    --train_start_date 2018-01-01 \
    --train_end_date 2022-12-31
```

### Workflow 2: Fine-Tune Next Month
```bash
python main.py \
    --policy HGAT \
    --baseline_checkpoint ./checkpoints/SmartFolio_2022-12.zip \
    --persistency_lambda 1e-4 \
    --resume_model_path ./checkpoints/SmartFolio_2022-12.zip \
    --train_start_date 2023-01-01 \
    --train_end_date 2023-01-31
```

### Workflow 3: Monthly Loop with Promotion
```bash
python main.py \
    --run_monthly_fine_tune \
    --baseline_checkpoint ./checkpoints/baseline.zip \
    --persistency_lambda 1e-4 \
    --promotion_min_sharpe 0.5 \
    --promotion_max_drawdown 0.2
```

---

## 📝 Notes

- Persistency is **disabled by default** (`lambda=0.0`)
- Baseline must be a valid SB3 `.zip` checkpoint
- Architecture must match between baseline and fine-tuned model
- Works with **all SmartFolio policies** (MLP, HGAT, custom)
- Compatible with **IRL training** and **standard PPO**
- **No breaking changes** to existing code

---

**Version:** 1.0  
**Last Updated:** November 28, 2025  
**Status:** ✅ Production Ready
