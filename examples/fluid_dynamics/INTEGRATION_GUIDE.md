# Using FluidDynamicsUNet with Your CFDBench Dataset

This guide shows how to integrate the `FluidDynamicsUNet` with your `FlowCastWrapperDataset`.

## Updated Model Signature

The model now accepts an `extra` dict for compatibility with existing training scripts:

```python
# New signature
model(x_t, t, extra)

# Where extra is:
extra = {
    'x_prev_1': tensor,  # (B, 2, H, W)
    'x_prev_2': tensor,  # (B, 2, H, W)
    'case_params': tensor  # (B, D) - optional
}
```

## Collate Function for Your Dataset

```python
import torch

def collate_fn(batch):
    """
    Convert FlowCastWrapperDataset output to format expected by training loop.

    Your dataset returns:
        - 'inputs_prev': X_{t-2}
        - 'inputs': X_{t-1}
        - 'label': X_{t}
        - 'case_params': dict of parameters

    Training loop expects:
        - samples: what to generate (the target)
        - labels: conditioning information
    """
    inputs_prev = torch.stack([item['inputs_prev'] for item in batch])
    inputs = torch.stack([item['inputs'] for item in batch])
    labels_target = torch.stack([item['label'] for item in batch])

    # Extract velocity components only (remove mask if present)
    # Adjust indices based on your data: if channels are [u, v, mask]
    inputs_prev = inputs_prev[:, :2, :, :]  # Keep only u, v
    inputs = inputs[:, :2, :, :]
    labels_target = labels_target[:, :2, :, :]

    # Convert case_params dict to tensor
    case_params_list = [item['case_params'] for item in batch]

    # Get keys in sorted order for consistency
    param_keys = sorted(case_params_list[0].keys())
    case_params_tensor = torch.tensor([
        [float(params[key]) for key in param_keys]
        for params in case_params_list
    ], dtype=torch.float32)

    # samples = what we're trying to generate (x_target)
    samples = labels_target

    # labels = conditioning info (everything else)
    labels = {
        'x_prev_1': inputs,
        'x_prev_2': inputs_prev,
        'case_params': case_params_tensor
    }

    return samples, labels
```

## Option 1: Use Existing train_loop.py (Minimal Changes)

The model signature is now compatible with the existing `train_loop.py`:

```python
from flow_matching.path import CondOTProbPath
from models.fluid_unet import FluidDynamicsUNet
from your_module import FlowCastWrapperDataset, CavityFlowAutoDataset
from torch.utils.data import DataLoader

# 1. Create dataset
base_dataset = CavityFlowAutoDataset(...)
wrapped_dataset = FlowCastWrapperDataset(base_dataset)

# 2. Auto-detect parameters
sample = wrapped_dataset[0]
num_case_params = len(sample['case_params'])
print(f"Detected {num_case_params} case parameters: {list(sample['case_params'].keys())}")

# 3. Create dataloader with custom collate
train_loader = DataLoader(
    wrapped_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
)

# 4. Create model
model = FluidDynamicsUNet(
    in_channels=2,  # u, v velocities
    out_channels=2,
    model_channels=128,
    num_case_params=num_case_params,
    use_fourier_conditioning=True,
).to(device)

# 5. Now use existing train_loop.py!
# The train loop will call: model(x_t, t, extra={'x_prev_1': ..., 'x_prev_2': ..., 'case_params': ...})
```

### Modify train_loop.py slightly:

```python
# In train_one_epoch function of train_loop.py
for samples, labels in data_loader:
    samples = samples.to(device)  # x_target

    # Unpack labels dict
    x_prev_1 = labels['x_prev_1'].to(device)
    x_prev_2 = labels['x_prev_2'].to(device)
    case_params = labels['case_params'].to(device)

    # Repack for model
    extra = {
        'x_prev_1': x_prev_1,
        'x_prev_2': x_prev_2,
        'case_params': case_params
    }

    # Flow matching
    t = torch.rand(samples.shape[0]).to(device)
    x_0 = torch.randn_like(samples)
    path_sample = path.sample(t=t, x_0=x_0, x_1=samples)

    # Forward pass
    with torch.cuda.amp.autocast():
        loss = torch.pow(model(path_sample.x_t, t, extra) - path_sample.dx_t, 2).mean()

    # Rest of training loop...
```

## Option 2: Use Custom train_fluid.py (Already Updated)

The `train_fluid.py` script is already set up for your use case:

```python
from models.fluid_unet import FluidDynamicsUNet
from data.fluid_dataset import FluidDynamicsDataset
# OR use your dataset:
from your_module import FlowCastWrapperDataset, CavityFlowAutoDataset

# Auto-detect from your data
base_dataset = CavityFlowAutoDataset(...)
wrapped_dataset = FlowCastWrapperDataset(base_dataset)
sample = wrapped_dataset[0]
num_case_params = len(sample['case_params'])

# Create model
model = FluidDynamicsUNet(
    num_case_params=num_case_params,
    use_fourier_conditioning=True,
).to(device)

# Training loop already handles the extra dict format
# See train_fluid.py for complete example
```

## Key Points

✅ **Model signature changed** to `forward(x_t, t, extra)` for compatibility
✅ **Internal processing unchanged** - still concatenates x_prev_1, x_prev_2 spatially
✅ **Collate function** converts your data format to (samples, labels) tuple
✅ **Auto-detect** number of case parameters from your dataset
✅ **Remove mask channel** if your data has [u, v, mask] format

## Quick Test

```python
# Test that your data works with the model
from models.fluid_unet import FluidDynamicsUNet

# Get one batch
batch = next(iter(DataLoader(wrapped_dataset, batch_size=4, collate_fn=collate_fn)))
samples, labels = batch

# Create model
model = FluidDynamicsUNet(num_case_params=len(labels['case_params'][0]))

# Test forward pass
import torch
x_t = torch.randn_like(samples)
t = torch.rand(samples.shape[0])
output = model(x_t, t, labels)

print(f"Input shape: {samples.shape}")
print(f"Output shape: {output.shape}")
print(f"Case params: {labels['case_params'].shape}")
# Expected output shapes match!
```

## Common Issues

### Issue: "KeyError: 'x_prev_1'"
**Solution**: Make sure your collate_fn returns labels as a dict with the right keys

### Issue: "Shape mismatch"
**Solution**: Check that you're removing the mask channel if present

### Issue: "Different number of case_params"
**Solution**: Ensure param_keys are sorted for consistency across batches

## Next Steps

1. ✅ Create your collate function based on the template above
2. ✅ Test with one batch to verify shapes
3. ✅ Run training with existing train_loop.py or train_fluid.py
4. ✅ Monitor loss - should decrease steadily

See `README.md` for complete training examples and hyperparameter tuning.
