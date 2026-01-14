"""
Quick demo script to test the fluid dynamics flow matching implementation.

This runs a minimal training loop on dummy data to verify everything works.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.fluid_unet import FluidDynamicsUNet
from data.fluid_dataset import FluidDynamicsDataset, create_dummy_fluid_data, normalize_vector_fields
from flow_matching.path import CondOTProbPath


def demo_training():
    """Run a quick training demo."""
    print("=" * 60)
    print("Flow Matching for Fluid Dynamics - Quick Demo")
    print("=" * 60)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Create small dummy dataset
    print("\n1. Creating dummy data...")
    vector_fields, case_params = create_dummy_fluid_data(
        num_cases=20,
        num_timesteps=15,
        height=64,
        width=64,
        num_params=3,
    )
    print(f"   Data shape: {vector_fields.shape}")
    print(f"   Case params shape: {case_params.shape}")

    # Normalize
    vector_fields, norm_stats = normalize_vector_fields(vector_fields)

    # Create dataset
    dataset = FluidDynamicsDataset(
        vector_fields=vector_fields,
        case_params=case_params,
        sequence_length=3,
    )
    print(f"   Dataset size: {len(dataset)} samples")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    # Create model
    print("\n2. Creating model...")
    model = FluidDynamicsUNet(
        in_channels=2,
        out_channels=2,
        model_channels=64,  # Small for demo
        channel_mult=(1, 2, 2),
        num_res_blocks=2,
        attention_resolutions=(2,),
        dropout=0.1,
        num_case_params=3,
        use_fourier_conditioning=True,
        num_fourier_freqs=8,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params:,}")

    # Create optimizer and path
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    path = CondOTProbPath()

    # Quick training loop
    print("\n3. Running training for 5 epochs...")
    model.train()

    for epoch in range(5):
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/5")
        for batch in pbar:
            # Get data
            x_target = batch['x_target'].to(device)
            x_prev_1 = batch['x_prev_1'].to(device)
            x_prev_2 = batch['x_prev_2'].to(device)
            case_params_batch = batch['case_params'].to(device)

            batch_size = x_target.shape[0]

            # Flow matching
            t = torch.rand(batch_size, device=device)
            x_0 = torch.randn_like(x_target)

            path_sample = path.sample(t=t, x_0=x_0, x_1=x_target)
            x_t = path_sample.x_t
            dx_t = path_sample.dx_t

            # Forward pass
            predicted_velocity = model(
                x_t=x_t,
                t=t,
                x_prev_1=x_prev_1,
                x_prev_2=x_prev_2,
                case_params=case_params_batch
            )

            # Loss
            loss = torch.nn.functional.mse_loss(predicted_velocity, dx_t)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track
            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches
        print(f"   Epoch {epoch+1} average loss: {avg_loss:.6f}")

    # Test inference
    print("\n4. Testing inference...")
    model.eval()

    with torch.no_grad():
        # Get a test sample
        test_batch = next(iter(dataloader))
        x_prev_1 = test_batch['x_prev_1'][:1].to(device)
        x_prev_2 = test_batch['x_prev_2'][:1].to(device)
        test_params = test_batch['case_params'][:1].to(device)

        # Sample from the model
        from flow_matching.solver import ODESolver
        import torch.nn as nn

        class ModelWrapper(nn.Module):
            def __init__(self, model, x_prev_1, x_prev_2, case_params):
                super().__init__()
                self.model = model
                self.x_prev_1 = x_prev_1
                self.x_prev_2 = x_prev_2
                self.case_params = case_params

            def forward(self, t, x):
                batch_size = x.shape[0]
                if t.dim() == 0:
                    t = t.repeat(batch_size)
                return self.model(x, t, self.x_prev_1, self.x_prev_2, self.case_params)

        wrapped_model = ModelWrapper(model, x_prev_1, x_prev_2, test_params)
        solver = ODESolver(wrapped_model)

        x_0 = torch.randn_like(x_prev_1)
        prediction = solver.sample(x_0, step_size=0.05)  # 20 steps

        print(f"   Input shape: {x_prev_1.shape}")
        print(f"   Prediction shape: {prediction.shape}")
        print(f"   Prediction mean: {prediction.mean().item():.4f}")
        print(f"   Prediction std: {prediction.std().item():.4f}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run full training: python train_fluid.py")
    print("  2. Check the README.md for more details")
    print("  3. Replace dummy data with your actual fluid dynamics data")
    print("=" * 60)


if __name__ == "__main__":
    demo_training()
