# Fluid Dynamics Prediction with Flow Matching

This example demonstrates how to use the flow_matching library for autoregressive fluid dynamics prediction with conditioning on previous states and case parameters.

## Overview

The implementation uses **Flow Matching** to learn the conditional distribution P(x_t | x_{t-1}, x_{t-2}, case_params) where:
- `x_t` is the velocity field at time t (shape: 2×64×64)
- `x_{t-1}`, `x_{t-2}` are previous states used for conditioning
- `case_params` are geometric/physical parameters (e.g., Reynolds number, cylinder radius)

### Key Features

1. **Autoregressive Conditioning**: The model uses states at t-1 and t-2 to predict t
2. **Case Parameter Conditioning**: Optional Fourier feature embeddings for continuous parameters
3. **Flow Matching Framework**: Uses continuous flow matching with conditional OT paths
4. **UNet Architecture**: Custom UNet with timestep and conditioning embeddings

## Architecture

### Model Components

```
FluidDynamicsUNet
├── Input: Concatenate [x_t, x_{t-1}, x_{t-2}]  (6 channels)
├── Timestep Embedding: Sinusoidal encoding of flow matching time t ∈ [0,1]
├── Case Parameter Embedding: Optional Fourier features
├── UNet Encoder: ResBlocks + Attention + Downsampling
├── UNet Middle: ResBlocks + Attention
├── UNet Decoder: ResBlocks + Attention + Upsampling (with skip connections)
└── Output: Predicted velocity field (2 channels)
```

### Flow Matching Training

For each training sample:
1. Sample timestep: `t ~ U[0, 1]`
2. Sample noise: `x_0 ~ N(0, I)`
3. Interpolate: `x_t = (1-t)·x_0 + t·x_1` where `x_1` is the target state
4. Compute target velocity: `dx_t = x_1 - x_0`
5. Predict velocity: `v_θ(x_t, t, x_{t-1}, x_{t-2}, params)`
6. Loss: `||v_θ - dx_t||²`

### Inference (Sampling)

To predict the next state:
1. Start from noise: `x_0 ~ N(0, I)`
2. Integrate ODE: `dx/dt = v_θ(x, t, x_{t-1}, x_{t-2}, params)` from t=0 to t=1
3. The result `x_1` is the predicted next state

## Installation

Make sure you have the flow_matching library installed:

```bash
cd /path/to/flow_matching
pip install -e .
```

Additional dependencies:
```bash
pip install matplotlib scikit-learn tqdm
```

## Usage

### Training

Train on dummy data (for demonstration):
```bash
python train_fluid.py \
    --num_cases 100 \
    --num_timesteps 50 \
    --num_case_params 3 \
    --model_channels 128 \
    --batch_size 16 \
    --num_epochs 100 \
    --lr 1e-4 \
    --use_fourier_conditioning \
    --save_dir ./checkpoints
```

Train on your own data:
```bash
python train_fluid.py \
    --data_path /path/to/your/data.pt \
    --num_case_params 5 \
    --model_channels 256 \
    --batch_size 32 \
    --num_epochs 200
```

**Data Format**: Your data file should contain:
```python
{
    'vector_fields': torch.Tensor,  # Shape: (N, T, 2, H, W)
                                    # N = number of cases
                                    # T = timesteps per case
                                    # 2 = velocity components (u, v)
                                    # H, W = spatial dimensions (64, 64)
    'case_params': torch.Tensor,    # Shape: (N, D)
                                    # D = number of case parameters
}
```

### Inference

Single-step prediction:
```bash
python inference.py \
    --checkpoint ./checkpoints/best_model.pt \
    --num_steps 50 \
    --num_predictions 1
```

Multi-step autoregressive prediction with visualization:
```bash
python inference.py \
    --checkpoint ./checkpoints/best_model.pt \
    --num_steps 50 \
    --num_predictions 10 \
    --visualize
```

### Programmatic Usage

```python
import torch
from models.fluid_unet import FluidDynamicsUNet
from inference import FluidPredictor

# Load model
checkpoint = torch.load('checkpoints/best_model.pt')
model = FluidDynamicsUNet(
    in_channels=2,
    out_channels=2,
    model_channels=128,
    num_case_params=3,
    use_fourier_conditioning=True,
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Create predictor
predictor = FluidPredictor(model, device='cuda')

# Prepare inputs
x_prev_1 = torch.randn(1, 2, 64, 64)  # State at t-1
x_prev_2 = torch.randn(1, 2, 64, 64)  # State at t-2
case_params = torch.tensor([[100.0, 0.3, 1.5]])  # [Reynolds, radius, velocity]

# Single prediction
next_state = predictor.predict_next_state(x_prev_1, x_prev_2, case_params)

# Autoregressive sequence
predictions = predictor.predict_sequence(
    x_prev_1, x_prev_2, case_params,
    num_predictions=10,
    num_steps=50
)
```

## Model Configuration

### Key Hyperparameters

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|-------------------|
| `model_channels` | Base channel count | 128 | 64-256 |
| `num_res_blocks` | Residual blocks per level | 2 | 2-4 |
| `dropout` | Dropout rate | 0.1 | 0.0-0.3 |
| `num_fourier_freqs` | Fourier feature frequencies | 16 | 8-32 |
| `learning_rate` | AdamW learning rate | 1e-4 | 1e-5 to 1e-3 |
| `num_steps` | ODE integration steps | 50 | 20-100 |

### Architecture Variants

For **smaller/faster models**:
```python
model = FluidDynamicsUNet(
    model_channels=64,
    channel_mult=(1, 2, 2),
    num_res_blocks=2,
    attention_resolutions=(2,),
)
```

For **larger/more accurate models**:
```python
model = FluidDynamicsUNet(
    model_channels=256,
    channel_mult=(1, 2, 3, 4),
    num_res_blocks=3,
    attention_resolutions=(2, 4),
    num_heads=8,
)
```

## Tips for Your Own Data

### 1. Data Preparation

- **Normalize your data**: Use `normalize_vector_fields()` to standardize
- **Temporal resolution**: Ensure sufficient temporal resolution (aim for smooth transitions)
- **Spatial resolution**: 64×64 is good for demonstration; increase for production
- **Sequence length**: At least 10-20 timesteps per case recommended

### 2. Case Parameters

Examples of useful parameters for flow past a cylinder:
- Reynolds number
- Cylinder radius or diameter
- Inlet velocity magnitude
- Angle of attack
- Fluid density/viscosity (if varying)

Use Fourier features for better embedding of continuous parameters!

### 3. Training Tips

- **Start with a small model** to verify your pipeline works
- **Monitor validation loss** - should decrease steadily
- **Skewed timesteps** (`--use_skewed_timesteps`) can help with difficult regions
- **Gradient clipping** may help with stability (add to optimizer)
- **Data augmentation**: Consider flips, rotations if appropriate

### 4. Evaluation

Beyond MSE, consider:
- **Physical constraints**: Conservation of mass, energy
- **Long-term stability**: Run 50+ step predictions
- **Vorticity/curl**: Check if predicted fields are physically reasonable
- **Spectral analysis**: Compare power spectra with ground truth

## Understanding Flow Matching for Physics

Flow matching learns a **generative model** of the data distribution. Unlike direct regression:

**Advantages:**
- Captures uncertainty in predictions
- Can generate diverse plausible futures
- More robust to noise in training data
- Better long-term stability in autoregressive rollouts

**Trade-offs:**
- Requires ODE solving at inference (slower)
- May need more training data
- Stochastic predictions (can be averaged for deterministic output)

For **deterministic predictions**, you can:
1. Use multiple samples and average them
2. Use a fixed random seed
3. Use a smaller noise scale in sampling

## Citation

If you use this code, please cite the flow matching paper:

```bibtex
@article{flowmatching2024,
  title={Flow Matching Guide and Code},
  author={...},
  journal={arXiv preprint arXiv:2412.06264},
  year={2024}
}
```

## License

This code is licensed under CC-BY-NC 4.0 (same as the flow_matching library).
