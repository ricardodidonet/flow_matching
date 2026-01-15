# Argparse Configuration Guide

This guide shows different ways to organize argparse arguments for cleaner code.

## Why Separate Configuration?

**Problems with inline argparse:**
- 50+ lines of argument definitions clutter your main script
- Hard to reuse arguments across train/eval/inference scripts
- Difficult to maintain and modify
- No clear grouping of related arguments

**Benefits of separation:**
- ✅ Clean, readable main script
- ✅ Reusable configuration across scripts
- ✅ Organized argument groups
- ✅ Easy to maintain and extend

## Approach 1: Separate Module (Recommended)

### Structure
```
examples/fluid_dynamics/
├── config.py              # All argument definitions
├── train_fluid_clean.py   # Clean training script
└── inference.py           # Inference script
```

### Usage

**Basic usage:**
```python
from config import get_train_parser

parser = get_train_parser()
args = parser.parse_args()

# Now use args
model = FluidDynamicsUNet(
    model_channels=args.model_channels,
    dropout=args.dropout,
    # ...
)
```

**With organized groups:**
```bash
python train_fluid_clean.py --help
```

Output shows organized groups:
```
Data Parameters:
  --data_path PATH       Path to fluid dynamics data
  --num_cases N          Number of simulation cases (default: 100)

Model Parameters:
  --model_channels N     Base channel count (default: 128)
  --dropout FLOAT        Dropout probability (default: 0.1)

Training Parameters:
  --batch_size N         Batch size (default: 16)
  --lr FLOAT             Learning rate (default: 0.0001)
```

### Running with custom arguments:
```bash
# Use defaults
python train_fluid_clean.py

# Override specific arguments
python train_fluid_clean.py \
    --model_channels 256 \
    --batch_size 32 \
    --num_epochs 200 \
    --lr 5e-5

# Use all custom settings
python train_fluid_clean.py \
    --data_path /path/to/data.pt \
    --model_channels 256 \
    --num_res_blocks 3 \
    --batch_size 32 \
    --num_epochs 200 \
    --lr 5e-5 \
    --conditioning_drop_prob 0.15
```

## Approach 2: Dataclass Configuration

More Pythonic, type-safe configuration:

```python
from config import TrainConfig
from dataclasses import asdict

# Create default config
config = TrainConfig()

# Modify as needed
config.model.model_channels = 256
config.training.batch_size = 32
config.training.lr = 5e-5

# Or from argparse
parser = get_train_parser()
args = parser.parse_args()
config = TrainConfig.from_args(args)

# Use config
model = FluidDynamicsUNet(
    model_channels=config.model.model_channels,
    dropout=config.model.dropout,
    num_case_params=config.data.num_case_params,
    use_fourier_conditioning=config.model.use_fourier_conditioning,
)
```

**Benefits:**
- Type hints and IDE autocomplete
- Clear structure with nested configs
- Easy to serialize (to JSON/YAML)
- Can set defaults programmatically

## Approach 3: Config Files (YAML/JSON)

For experiments, save/load full configs:

**Create config file `experiment1.json`:**
```json
{
  "data_path": null,
  "num_cases": 100,
  "model_channels": 256,
  "num_res_blocks": 3,
  "batch_size": 32,
  "num_epochs": 200,
  "lr": 5e-5,
  "conditioning_drop_prob": 0.15
}
```

**Load and use:**
```python
import json
import argparse

# Load config file
with open('experiment1.json', 'r') as f:
    config_dict = json.load(f)

# Create parser with defaults from file
parser = get_train_parser()
parser.set_defaults(**config_dict)

# Can still override from command line
args = parser.parse_args()

# Or programmatically override
parser.set_defaults(**config_dict)
args = parser.parse_args(['--batch_size', '64'])  # Override just batch_size
```

**Usage:**
```bash
# Load config, train with it
python train_fluid_clean.py --config experiment1.json

# Load config but override some values
python train_fluid_clean.py --config experiment1.json --lr 1e-4 --batch_size 64
```

## Approach 4: Hydra (Advanced)

For serious ML projects, use [Hydra](https://hydra.cc/):

**Config file `config.yaml`:**
```yaml
data:
  data_path: null
  num_cases: 100
  num_timesteps: 50

model:
  model_channels: 128
  num_res_blocks: 2
  dropout: 0.1

training:
  batch_size: 16
  num_epochs: 100
  lr: 1e-4
```

**Script:**
```python
import hydra
from omegaconf import DictConfig

@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    model = FluidDynamicsUNet(
        model_channels=cfg.model.model_channels,
        dropout=cfg.model.dropout,
    )
    # ...

if __name__ == "__main__":
    main()
```

**Usage:**
```bash
# Use default config
python train.py

# Override from command line
python train.py model.model_channels=256 training.batch_size=32

# Use different config file
python train.py --config-name experiment1
```

## Comparison

| Approach | Complexity | Flexibility | Best For |
|----------|-----------|-------------|----------|
| Separate module | Low | Medium | Small-medium projects |
| Dataclass | Medium | High | Type-safe configs |
| Config files | Medium | High | Experiment tracking |
| Hydra | High | Very High | Large ML projects |

## Recommended Setup for Your Project

**Start with Approach 1 (Separate Module):**

```python
# config.py - Organize arguments
from config import get_train_parser, get_inference_parser

# train_fluid_clean.py - Use clean config
parser = get_train_parser()
args = parser.parse_args()

# Save config for reproducibility
from config import save_config_to_file
save_config_to_file(args, "checkpoints/config.json")
```

**Upgrade to Approach 2 (Dataclass) when needed:**
- More structure needed
- Want type checking
- Multiple scripts sharing config

**Upgrade to Approach 3/4 when:**
- Running many experiments
- Need to reproduce results
- Hyperparameter sweeps

## Practical Examples

### Example 1: Quick Experiment

```bash
# Try larger model
python train_fluid_clean.py --model_channels 256 --num_res_blocks 3

# More aggressive dropout
python train_fluid_clean.py --dropout 0.3

# Longer training
python train_fluid_clean.py --num_epochs 500 --lr 5e-5
```

### Example 2: Save Best Config

```python
# In your training script
if val_loss < best_val_loss:
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'args': vars(args),  # Save full config!
    }
    torch.save(checkpoint, 'best_model.pt')

# Later, load and reproduce
checkpoint = torch.load('best_model.pt')
saved_args = checkpoint['args']
print(f"Best model used: lr={saved_args['lr']}, batch_size={saved_args['batch_size']}")
```

### Example 3: Multiple Configs

Create different config files for different experiments:

**`configs/baseline.json`:**
```json
{
  "model_channels": 128,
  "batch_size": 16,
  "lr": 1e-4
}
```

**`configs/large_model.json`:**
```json
{
  "model_channels": 256,
  "num_res_blocks": 3,
  "batch_size": 32,
  "lr": 5e-5
}
```

**Run experiments:**
```bash
python train_fluid_clean.py --config configs/baseline.json
python train_fluid_clean.py --config configs/large_model.json
```

## Tips and Best Practices

### 1. Always save configuration
```python
save_config_to_file(args, save_dir / "config.json")
```

### 2. Use argument groups
Makes `--help` output readable:
```python
group = parser.add_argument_group('Model Parameters')
group.add_argument(...)
```

### 3. Set good defaults
```python
parser.add_argument("--batch_size", type=int, default=16,
                    help="Batch size (default: 16)")
```

### 4. Use `action="store_true"` for flags
```python
parser.add_argument("--use_fourier_conditioning", action="store_true",
                    help="Enable Fourier features")
```

### 5. Validate arguments
```python
args = parser.parse_args()

# Validate
assert args.batch_size > 0, "batch_size must be positive"
assert 0 <= args.dropout < 1, "dropout must be in [0, 1)"
```

### 6. Use `formatter_class` for better help
```python
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Shows defaults
)
```

## Next Steps

1. ✅ Use `config.py` for your project (already created!)
2. ✅ Update `train_fluid.py` to import from `config`
3. Try running with different arguments
4. Save your best configurations
5. Consider dataclass approach for type safety

## Summary

**Quick wins:**
- Move args to `config.py` ✅
- Use argument groups ✅
- Save configs with checkpoints ✅

**For serious projects:**
- Use dataclasses for type safety
- Create config files for experiments
- Consider Hydra for complex setups
