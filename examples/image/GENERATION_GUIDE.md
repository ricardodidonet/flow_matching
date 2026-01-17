# Image Generation Guide

This guide shows you how to generate images from your trained flow matching model.

## Quick Start

```bash
cd /data/flow_matching/examples/image

# Basic: Generate 64 random samples
python generate_samples.py --checkpoint output_dir/checkpoint-100.pth --num_samples 64

# With EMA weights (better quality if you trained with --use_ema)
python generate_samples.py --checkpoint output_dir/checkpoint-100.pth --num_samples 64 --use_ema
```

## Examples

### 1. Generate Specific Class

```bash
# Generate 64 dogs (class 5)
python generate_samples.py \
    --checkpoint output_dir/checkpoint-100.pth \
    --num_samples 64 \
    --label 5 \
    --use_ema

# CIFAR-10 classes:
# 0: airplane, 1: automobile, 2: bird, 3: cat, 4: deer
# 5: dog, 6: frog, 7: horse, 8: ship, 9: truck
```

### 2. Use Classifier-Free Guidance

```bash
# Higher cfg_scale = stronger conditioning = more typical examples
python generate_samples.py \
    --checkpoint output_dir/checkpoint-100.pth \
    --num_samples 64 \
    --label 3 \
    --cfg_scale 0.5 \
    --use_ema

# Try different scales:
# --cfg_scale 0.0   # No guidance (diverse but may be less class-typical)
# --cfg_scale 0.3   # Mild guidance (good balance)
# --cfg_scale 0.5   # Medium guidance (clearer class features)
# --cfg_scale 1.0   # Strong guidance (very typical, may be too smooth)
```

### 3. Generate All Classes

```bash
# Generate 80 samples (8 per class)
python generate_samples.py \
    --checkpoint output_dir/checkpoint-100.pth \
    --num_samples 80 \
    --all_classes \
    --cfg_scale 0.3 \
    --use_ema
```

### 4. Save Individual Images

```bash
# Save both grid and individual PNG files
python generate_samples.py \
    --checkpoint output_dir/checkpoint-100.pth \
    --num_samples 100 \
    --save_individual \
    --output_dir my_generated_images \
    --use_ema
```

### 5. High-Quality Generation

```bash
# Use more ODE steps for better quality (slower)
python generate_samples.py \
    --checkpoint output_dir/checkpoint-100.pth \
    --num_samples 64 \
    --label 3 \
    --cfg_scale 0.5 \
    --ode_method dopri5 \
    --nfe 100 \
    --use_ema

# Or use EDM schedule (recommended)
python generate_samples.py \
    --checkpoint output_dir/checkpoint-100.pth \
    --num_samples 64 \
    --edm_schedule \
    --nfe 50 \
    --use_ema
```

### 6. Fast Generation (Lower Quality)

```bash
# Use fewer steps for quick results
python generate_samples.py \
    --checkpoint output_dir/checkpoint-100.pth \
    --num_samples 64 \
    --ode_method euler \
    --step_size 0.1 \
    --use_ema
```

## Parameter Guide

### Model Parameters
- `--checkpoint`: Path to your trained model checkpoint (required)
- `--use_ema`: Use EMA weights (use this if you trained with `--use_ema`)
- `--dataset`: Dataset type (default: cifar10)

### Generation Parameters
- `--num_samples`: How many images to generate (default: 64)
- `--label`: Specific class to generate (0-9 for CIFAR-10, or None for random)
- `--all_classes`: Generate equal samples for all classes
- `--cfg_scale`: Classifier-free guidance strength (0.0 to 2.0, default: 0.0)

### ODE Solver Parameters
- `--ode_method`: Solver algorithm
  - `euler`: Fastest, lowest quality
  - `midpoint`: Good balance (default)
  - `heun2`: Better quality
  - `dopri5`: Best quality, slowest
- `--step_size`: Step size for fixed-step solvers (default: 0.05)
- `--nfe`: Number of function evaluations for adaptive solvers (default: 50)
- `--edm_schedule`: Use EDM time discretization (recommended for best quality)

### Output Parameters
- `--output_dir`: Where to save images (default: ./generated_samples)
- `--save_individual`: Save individual PNG files (not just grid)
- `--nrow`: Images per row in grid (default: 8)
- `--seed`: Random seed for reproducibility (default: 42)

## ODE Solver Comparison

| Method | Speed | Quality | NFE (typical) | When to Use |
|--------|-------|---------|---------------|-------------|
| `euler` | ⚡⚡⚡ | ⭐ | 20 | Quick previews |
| `midpoint` | ⚡⚡ | ⭐⭐ | 20 | Default choice |
| `heun2` | ⚡⚡ | ⭐⭐⭐ | 50 | Good quality |
| `dopri5` | ⚡ | ⭐⭐⭐⭐ | 100+ | Best quality |
| `midpoint + edm_schedule` | ⚡⚡ | ⭐⭐⭐⭐ | 50 | Recommended |

## Output Files

```
generated_samples/
├── samples_grid.png              # Grid of all samples
├── generation_summary.txt         # Class distribution info
└── individual/                    # (if --save_individual)
    ├── sample_0000_class5_dog.png
    ├── sample_0001_class5_dog.png
    └── ...
```

## Tips

1. **Always use `--use_ema`** if your model was trained with EMA (better quality)

2. **For best quality:**
   ```bash
   python generate_samples.py \
       --checkpoint <checkpoint> \
       --edm_schedule --nfe 50 \
       --cfg_scale 0.3 \
       --use_ema
   ```

3. **For quick experiments:**
   ```bash
   python generate_samples.py \
       --checkpoint <checkpoint> \
       --ode_method euler --step_size 0.2 \
       --num_samples 16
   ```

4. **To compare different cfg_scale values:**
   ```bash
   for cfg in 0.0 0.3 0.5 1.0; do
       python generate_samples.py \
           --checkpoint <checkpoint> \
           --label 3 --num_samples 64 \
           --cfg_scale $cfg \
           --output_dir results_cfg${cfg} \
           --use_ema
   done
   ```

5. **Generate samples for all your checkpoints:**
   ```bash
   for ckpt in output_dir/checkpoint-*.pth; do
       epoch=$(basename $ckpt .pth | cut -d'-' -f2)
       python generate_samples.py \
           --checkpoint $ckpt \
           --num_samples 64 \
           --output_dir samples_epoch${epoch} \
           --use_ema
   done
   ```

## Troubleshooting

**Error: "No module named 'flow_matching'"**
```bash
# Make sure you're in the right conda environment
conda activate flow_matching
cd /data/flow_matching/examples/image
```

**Error: "No model weights found in checkpoint"**
```bash
# Try without --use_ema if you didn't train with EMA
python generate_samples.py --checkpoint <checkpoint> --num_samples 64
```

**Out of memory error**
```bash
# Reduce batch size by generating fewer samples at once
python generate_samples.py --checkpoint <checkpoint> --num_samples 16

# Or use CPU (slower but no memory limits)
python generate_samples.py --checkpoint <checkpoint> --device cpu
```

**Images look bad**
- Make sure you're using a checkpoint from late in training (e.g., epoch 100+)
- Use `--use_ema` if available
- Try `--cfg_scale 0.3` to 0.5
- Use `--edm_schedule` with `--nfe 50`
