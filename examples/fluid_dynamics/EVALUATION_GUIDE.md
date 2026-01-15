# Evaluation Guide for Fluid Dynamics Model

This guide explains the evaluation system adapted from the image generation `eval_loop.py`.

## Overview

The evaluation system (`eval_fluid.py`) provides:

1. **Single-step prediction accuracy** - How well the model predicts the next frame
2. **Autoregressive rollout** - Long-term prediction stability
3. **Physical metrics** - MSE, MAE, relative error, component-wise errors
4. **Classifier-free guidance** - Control prediction fidelity via case parameters
5. **Visualizations** - Compare predictions vs ground truth

---

## Key Components

### 1. CFGScaledFluidModel

Wrapper for classifier-free guidance on case parameters (similar to class labels in images).

**How it works:**
```python
# With guidance (cfg_scale > 0)
conditional = model(x, t, x_prev_1, x_prev_2, case_params=params)      # With params
unconditional = model(x, t, x_prev_1, x_prev_2, case_params=None)      # Without params
result = unconditional + cfg_scale * (conditional - unconditional)

# No guidance (cfg_scale = 0)
result = model(x, t, x_prev_1, x_prev_2, case_params=params)
```

**CFG Scale Effects:**
- `cfg_scale = 0.0`: Standard conditional prediction
- `cfg_scale = 0.5`: Mild guidance (slight adherence to case params)
- `cfg_scale = 1.0`: Moderate guidance
- `cfg_scale = 1.5`: Strong guidance (predictions follow case params closely)
- `cfg_scale = 2.0`: Very strong guidance (may over-fit to params)

### 2. compute_physical_metrics()

Computes multiple error metrics between predicted and ground truth:

```python
metrics = {
    'mse': Mean Squared Error (overall),
    'mae': Mean Absolute Error,
    'relative_error': ||pred - target|| / ||target||,
    'u_mse': MSE for u-component only,
    'v_mse': MSE for v-component only,
    'magnitude_mse': MSE of velocity magnitudes,
}
```

### 3. single_step_prediction()

Generates one prediction: Given states at t-2 and t-1, predict state at t.

**Process:**
1. Start from Gaussian noise: `x_0 ~ N(0, I)`
2. Integrate ODE: `dx/dt = model(x, t, x_{t-1}, x_{t-2}, params)`
3. Return final state at t=1

### 4. autoregressive_rollout()

Performs multi-step predictions where each prediction becomes input for the next.

**Process:**
```
Initial: [x_{t-2}, x_{t-1}]

Step 1: Predict x_t    using [x_{t-2}, x_{t-1}]
Step 2: Predict x_{t+1} using [x_{t-1}, x_t]
Step 3: Predict x_{t+2} using [x_t, x_{t+1}]
...
```

**Key metric:** How quickly do errors accumulate over time?

### 5. visualize_prediction()

Creates side-by-side comparison plots:
- Ground truth vs predicted (U, V, magnitude)
- Color-coded velocity fields

---

## Usage

### Basic Evaluation

```bash
# Evaluate with default settings
python run_eval.py --checkpoint checkpoints/best_model.pt

# Evaluate with guidance
python run_eval.py \
    --checkpoint checkpoints/best_model.pt \
    --cfg_scale 1.5 \
    --num_ode_steps 50
```

### Programmatic Usage

```python
from eval_fluid import eval_fluid_model
from torch.utils.data import DataLoader

# Load model and data
model = FluidDynamicsUNet(...)
model.load_state_dict(checkpoint['model_state_dict'])
val_loader = DataLoader(val_dataset, batch_size=16)

# Run evaluation
metrics = eval_fluid_model(
    model=model,
    data_loader=val_loader,
    device=device,
    epoch=50,
    cfg_scale=1.0,
    num_ode_steps=50,
    num_rollout_steps=10,
    output_dir=Path("eval_results"),
)

print(f"MSE: {metrics['eval_mse']:.6f}")
print(f"Rollout error: {metrics['eval_rollout_error_mean']:.6f}")
```

---

## Metrics Explained

### Single-Step Metrics

| Metric | What It Measures | Good Value |
|--------|------------------|------------|
| `eval_mse` | Overall prediction error | < 0.01 |
| `eval_mae` | Average absolute error | < 0.05 |
| `eval_relative_error` | Error relative to magnitude | < 0.1 |
| `eval_u_mse` | Error in u-component | < 0.01 |
| `eval_v_mse` | Error in v-component | < 0.01 |
| `eval_magnitude_mse` | Error in velocity magnitude | < 0.01 |

### Rollout Metrics

| Metric | What It Measures | Good Value |
|--------|------------------|------------|
| `eval_rollout_error_mean` | Average error over rollout | < 0.05 |
| `eval_rollout_error_final` | Error at final step | < 0.1 |

### Computational Cost

| Metric | What It Measures | Notes |
|--------|------------------|-------|
| `eval_nfe_per_sample` | Function evaluations per prediction | Lower = faster |

**Typical values:**
- `num_ode_steps=20`: NFE ≈ 20-25
- `num_ode_steps=50`: NFE ≈ 50-60
- `num_ode_steps=100`: NFE ≈ 100-120

Trade-off: More steps = better accuracy but slower.

---

## Comparison: Images vs Fluid Dynamics

| Aspect | eval_loop.py (Images) | eval_fluid.py (Fluid) |
|--------|----------------------|----------------------|
| **Main metric** | FID score | MSE, Physical metrics |
| **Conditioning** | Class labels (0-999) | Case parameters (continuous) |
| **CFG target** | Class labels | Case parameters |
| **Temporal** | Single-frame generation | Autoregressive sequence |
| **Visualizations** | Image grid | Velocity field plots |
| **Quality check** | FID < 10 is good | MSE < 0.01 is good |

---

## Interpreting Results

### Good Model Signs

✅ **Low single-step error** (MSE < 0.01)
- Model accurately predicts immediate next state

✅ **Stable rollout** (error doesn't explode)
- Rollout error at step 10 ≈ 2-3× error at step 1

✅ **Balanced component errors** (u_mse ≈ v_mse)
- Model doesn't favor one velocity component

✅ **Low NFE** (< 60 for good accuracy)
- Efficient sampling

### Problem Signs

❌ **High single-step error** (MSE > 0.1)
- Model hasn't learned the dynamics well
- Solution: Train longer, check data quality

❌ **Exploding rollout error** (error × 10 after few steps)
- Model is unstable in autoregressive mode
- Solution: Increase dropout, add noise to training

❌ **Unbalanced errors** (u_mse >> v_mse or vice versa)
- Model biased toward one component
- Solution: Check data normalization

---

## CFG Scale Tuning

Run evaluation with different CFG scales to find optimal:

```bash
# No guidance
python run_eval.py --checkpoint model.pt --cfg_scale 0.0

# Mild guidance
python run_eval.py --checkpoint model.pt --cfg_scale 0.5

# Moderate guidance
python run_eval.py --checkpoint model.pt --cfg_scale 1.0

# Strong guidance
python run_eval.py --checkpoint model.pt --cfg_scale 1.5
```

**Expected behavior:**

| CFG Scale | Effect | Best For |
|-----------|--------|----------|
| 0.0 | Standard prediction | General cases |
| 0.5-1.0 | Subtle conditioning boost | Similar to training |
| 1.5-2.0 | Strong conditioning | Specific parameter values |
| > 2.0 | Over-conditioning | Usually too strong |

**Plot MSE vs CFG scale to find optimum!**

---

## Autoregressive Rollout Analysis

The rollout error plot shows how errors accumulate:

```
Good model:
  Step 1: MSE = 0.005
  Step 5: MSE = 0.015  (3× initial)
  Step 10: MSE = 0.025 (5× initial)

Bad model:
  Step 1: MSE = 0.005
  Step 5: MSE = 0.080  (16× initial)
  Step 10: MSE = 0.300 (60× initial)
```

**Ideal behavior:** Linear or sub-linear error growth

**Problem:** Exponential error growth → model is unstable

---

## Extending the Evaluation

### Add Custom Metrics

```python
def compute_physical_metrics(predicted, target):
    metrics = {}

    # Standard metrics
    metrics['mse'] = torch.nn.functional.mse_loss(predicted, target).item()

    # Add custom: Check divergence-free constraint
    # ∇·u = ∂u/∂x + ∂v/∂y should ≈ 0
    u = predicted[:, 0]
    v = predicted[:, 1]
    du_dx = torch.gradient(u, dim=2)[0]
    dv_dy = torch.gradient(v, dim=1)[0]
    divergence = torch.abs(du_dx + dv_dy).mean()
    metrics['divergence'] = divergence.item()

    # Add custom: Energy conservation
    pred_energy = (predicted**2).sum(dim=1).mean()
    target_energy = (target**2).sum(dim=1).mean()
    metrics['energy_error'] = torch.abs(pred_energy - target_energy).item()

    return metrics
```

### Conditional Evaluation

Evaluate separately for different parameter ranges:

```python
# Evaluate low Reynolds number cases
low_re_indices = case_params[:, 0] < 1000
low_re_metrics = eval_fluid_model(
    model, low_re_dataloader, ...
)

# Evaluate high Reynolds number cases
high_re_indices = case_params[:, 0] >= 1000
high_re_metrics = eval_fluid_model(
    model, high_re_dataloader, ...
)

print(f"Low Re MSE: {low_re_metrics['eval_mse']:.6f}")
print(f"High Re MSE: {high_re_metrics['eval_mse']:.6f}")
```

---

## Output Structure

After running evaluation:

```
eval_results/
├── eval_results.txt           # Text summary of all metrics
├── snapshots/                 # Single-step predictions
│   ├── epoch_50_batch_0.png
│   ├── epoch_50_batch_1.png
│   └── ...
└── rollouts/                  # Autoregressive rollout plots
    └── rollout_error_epoch_50.png
```

---

## Tips

1. **Always evaluate with multiple CFG scales** to find the sweet spot
2. **Check rollout stability** - this indicates real-world usability
3. **Compare to baseline** - Simple linear extrapolation, persistence model
4. **Visualize failures** - Look at worst-case predictions to understand limits
5. **Track NFE** - Balance accuracy vs computational cost

---

## Next Steps

1. Run evaluation on your trained model
2. Try different CFG scales
3. Analyze rollout stability
4. Compare to your original (non-flow-matching) approach
5. Tune based on which metric matters most for your application

For questions, see `ARGPARSE_GUIDE.md` for configuration and `README.md` for training details.
