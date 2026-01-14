# Quick Start Guide

## 1. Setup Environment

```bash
# Install the flow_matching library
cd /path/to/flow_matching
pip install -e .

# Install additional dependencies
pip install matplotlib scikit-learn tqdm
```

## 2. Verify Installation

Run the quick demo to verify everything works:

```bash
cd examples/fluid_dynamics
python demo.py
```

This will:
- Create dummy fluid dynamics data
- Train a small model for 5 epochs
- Test inference with the trained model

Expected output: Training should complete in 1-2 minutes with decreasing loss.

## 3. Prepare Your Data

Your data should be in the format:

```python
import torch

# Vector fields: (num_cases, num_timesteps, 2, height, width)
# - num_cases: number of different simulations
# - num_timesteps: temporal resolution per simulation
# - 2: velocity components (u, v)
# - height, width: spatial resolution (e.g., 64x64)
vector_fields = torch.randn(100, 50, 2, 64, 64)

# Case parameters: (num_cases, num_params)
# - num_params: geometric/physical parameters
# Example: [Reynolds_number, cylinder_radius, inlet_velocity]
case_params = torch.randn(100, 3)

# Save your data
torch.save({
    'vector_fields': vector_fields,
    'case_params': case_params
}, 'my_fluid_data.pt')
```

## 4. Train on Your Data

```bash
python train_fluid.py \
    --data_path my_fluid_data.pt \
    --num_case_params 3 \
    --model_channels 128 \
    --batch_size 16 \
    --num_epochs 100 \
    --lr 1e-4 \
    --save_dir ./checkpoints
```

## 5. Run Inference

```bash
python inference.py \
    --checkpoint ./checkpoints/best_model.pt \
    --num_predictions 10 \
    --visualize
```

This will:
- Load the trained model
- Perform autoregressive predictions
- Generate visualizations (saved as `fluid_predictions.png`)

## Common Issues

### Issue: "ModuleNotFoundError: No module named 'torch'"

**Solution**: Make sure PyTorch is installed:
```bash
pip install torch torchvision
```

### Issue: "CUDA out of memory"

**Solutions**:
1. Reduce batch size: `--batch_size 8`
2. Reduce model size: `--model_channels 64`
3. Use CPU: `--device cpu`

### Issue: Loss is not decreasing

**Possible causes**:
1. Learning rate too high/low - try `--lr 5e-5` or `--lr 5e-4`
2. Data not normalized - the code does this automatically, but verify your raw data is reasonable
3. Too few training samples - need at least ~50 cases with ~20 timesteps each

## Next Steps

1. Read the full [README.md](README.md) for detailed documentation
2. Adjust model architecture in `models/fluid_unet.py`
3. Modify training parameters in `train_fluid.py`
4. Customize the probability path (e.g., use different schedulers)

## Architecture Overview

```
Training:
  Data → Normalize → Dataset → DataLoader
                                    ↓
  FluidDynamicsUNet ← Flow Matching Loss
                                    ↓
                              Optimizer → Checkpoints

Inference:
  Checkpoint → FluidDynamicsUNet → ODE Solver → Predictions
                        ↑
          [x_{t-1}, x_{t-2}, params]
```

## Key Files

- `models/fluid_unet.py` - UNet architecture with conditioning
- `data/fluid_dataset.py` - Dataset and data utilities
- `train_fluid.py` - Main training script
- `inference.py` - Inference and autoregressive prediction
- `demo.py` - Quick verification script

## Getting Help

If you encounter issues:
1. Check the error message carefully
2. Review the README.md for detailed explanations
3. Make sure your data format matches the expected format
4. Try the demo.py script first to verify installation
5. Check the flow_matching library documentation at the repository root
