"""
Training script for fluid dynamics prediction using flow matching.

This script demonstrates how to use the flow_matching library for autoregressive
fluid dynamics prediction with conditioning on previous states and case parameters.
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Flow matching imports
from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver

# Local imports
from models.fluid_unet import FluidDynamicsUNet
from data.fluid_dataset import (
    FluidDynamicsDataset,
    create_dummy_fluid_data,
    normalize_vector_fields,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train fluid dynamics flow matching model")

    # Data parameters
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to fluid dynamics data (if None, uses dummy data)")
    parser.add_argument("--num_cases", type=int, default=100,
                        help="Number of simulation cases (for dummy data)")
    parser.add_argument("--num_timesteps", type=int, default=50,
                        help="Number of timesteps per case (for dummy data)")
    parser.add_argument("--num_case_params", type=int, default=3,
                        help="Number of case parameters")

    # Model parameters
    parser.add_argument("--model_channels", type=int, default=128,
                        help="Base channel count for UNet")
    parser.add_argument("--num_res_blocks", type=int, default=2,
                        help="Number of residual blocks per resolution")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability")
    parser.add_argument("--use_fourier_conditioning", action="store_true", default=True,
                        help="Use Fourier features for case parameters")
    parser.add_argument("--num_fourier_freqs", type=int, default=16,
                        help="Number of Fourier frequencies")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--train_split", type=float, default=0.8,
                        help="Fraction of data for training")

    # Flow matching parameters
    parser.add_argument("--use_skewed_timesteps", action="store_true",
                        help="Use skewed timestep sampling (focus on difficult regions)")
    parser.add_argument("--conditioning_drop_prob", type=float, default=0.1,
                        help="Probability of dropping conditioning for classifier-free guidance (default: 0.1)")

    # System parameters
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Logging interval in batches")

    return parser.parse_args()


def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    """
    Sample timesteps with skewed distribution to focus on difficult regions.
    This puts more emphasis on t near 0 and 1.
    """
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    time = 1 / (1 + sigma)
    time = torch.clip(time, min=0.0001, max=1.0)
    return time


def train_epoch(model, dataloader, optimizer, path, device, args, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        x_target = batch['x_target'].to(device)  # (B, 2, H, W)
        x_prev_1 = batch['x_prev_1'].to(device)
        x_prev_2 = batch['x_prev_2'].to(device)

        case_params = None
        if 'case_params' in batch:
            case_params = batch['case_params'].to(device)

        batch_size = x_target.shape[0]

        # Sample random timesteps for flow matching
        if args.use_skewed_timesteps:
            t = skewed_timestep_sample(batch_size, device)
        else:
            t = torch.rand(batch_size, device=device)

        # Sample noise for x_0 (Gaussian noise)
        x_0 = torch.randn_like(x_target)

        # Sample from the probability path: x_t = (1-t)*x_0 + t*x_1
        # where x_1 is the target state
        path_sample = path.sample(t=t, x_0=x_0, x_1=x_target)
        x_t = path_sample.x_t  # Noisy interpolation
        dx_t = path_sample.dx_t  # Target velocity: x_1 - x_0

        # Classifier-free guidance: randomly drop case_params
        # Keep x_prev_1, x_prev_2 (essential for temporal prediction)
        # Drop case_params (optional geometric/physical conditioning)
        if torch.rand(1).item() < args.conditioning_drop_prob:
            extra = {
                'x_prev_1': x_prev_1,
                'x_prev_2': x_prev_2,
                'case_params': None  # Dropped for unconditional training
            }
        else:
            extra = {
                'x_prev_1': x_prev_1,
                'x_prev_2': x_prev_2,
                'case_params': case_params
            }

        # Forward pass: predict velocity field conditioned on previous states
        predicted_velocity = model(x_t, t, extra)

        # Flow matching loss: L2 between predicted and target velocity
        loss = torch.nn.functional.mse_loss(predicted_velocity, dx_t)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        total_loss += loss.item()
        num_batches += 1

        if batch_idx % args.log_interval == 0:
            pbar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def validate(model, dataloader, path, device, args):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Validation"):
        x_target = batch['x_target'].to(device)
        x_prev_1 = batch['x_prev_1'].to(device)
        x_prev_2 = batch['x_prev_2'].to(device)

        case_params = None
        if 'case_params' in batch:
            case_params = batch['case_params'].to(device)

        batch_size = x_target.shape[0]

        # Use fixed timesteps for validation
        t = torch.rand(batch_size, device=device)

        x_0 = torch.randn_like(x_target)
        path_sample = path.sample(t=t, x_0=x_0, x_1=x_target)
        x_t = path_sample.x_t
        dx_t = path_sample.dx_t

        extra = {
            'x_prev_1': x_prev_1,
            'x_prev_2': x_prev_2,
            'case_params': case_params
        }
        predicted_velocity = model(x_t, t, extra)

        loss = torch.nn.functional.mse_loss(predicted_velocity, dx_t)
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def sample_prediction(model, x_prev_1, x_prev_2, case_params, device, num_steps=50, guidance_scale=1.0):
    """
    Generate a prediction by sampling from the learned flow with optional classifier-free guidance.

    This uses the ODE solver to integrate the learned velocity field
    from noise (t=0) to data (t=1).

    Args:
        model: Trained model
        x_prev_1: Previous state at t-1, shape (B, 2, H, W)
        x_prev_2: Previous state at t-2, shape (B, 2, H, W)
        case_params: Case parameters, shape (B, D)
        device: Device
        num_steps: Number of ODE integration steps
        guidance_scale: Classifier-free guidance scale (1.0 = no guidance, >1.0 = stronger conditioning)

    Returns:
        Predicted next state, shape (B, 2, H, W)
    """
    model.eval()

    # Create a wrapper for the model that has the right signature for ODESolver
    class ModelWrapper(nn.Module):
        def __init__(self, model, x_prev_1, x_prev_2, case_params, guidance_scale):
            super().__init__()
            self.model = model
            self.x_prev_1 = x_prev_1
            self.x_prev_2 = x_prev_2
            self.case_params = case_params
            self.guidance_scale = guidance_scale

        def forward(self, t, x):
            # t is a scalar, we need to broadcast to batch size
            batch_size = x.shape[0]
            if t.dim() == 0:
                t = t.repeat(batch_size)

            if self.guidance_scale == 1.0 or self.case_params is None:
                # No guidance or no case params - standard prediction
                extra = {
                    'x_prev_1': self.x_prev_1,
                    'x_prev_2': self.x_prev_2,
                    'case_params': self.case_params
                }
                return self.model(x, t, extra)
            else:
                # Classifier-free guidance
                # Conditional prediction
                extra_cond = {
                    'x_prev_1': self.x_prev_1,
                    'x_prev_2': self.x_prev_2,
                    'case_params': self.case_params
                }
                v_cond = self.model(x, t, extra_cond)

                # Unconditional prediction (drop case_params)
                extra_uncond = {
                    'x_prev_1': self.x_prev_1,
                    'x_prev_2': self.x_prev_2,
                    'case_params': None
                }
                v_uncond = self.model(x, t, extra_uncond)

                # Guided velocity: v = v_uncond + guidance_scale * (v_cond - v_uncond)
                return v_uncond + self.guidance_scale * (v_cond - v_uncond)

    wrapped_model = ModelWrapper(model, x_prev_1, x_prev_2, case_params, guidance_scale)

    # Create ODE solver
    solver = ODESolver(wrapped_model)

    # Start from Gaussian noise
    x_0 = torch.randn_like(x_prev_1)

    # Solve ODE from t=0 to t=1
    x_1 = solver.sample(x_0, step_size=1.0 / num_steps)

    return x_1


def main():
    args = parse_args()

    # Set up device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load or create data
    if args.data_path is not None:
        logger.info(f"Loading data from {args.data_path}")
        # TODO: Implement custom data loading
        # data = torch.load(args.data_path)
        # vector_fields = data['vector_fields']
        # case_params = data['case_params']
        raise NotImplementedError("Custom data loading not yet implemented")
    else:
        logger.info("Creating dummy data for demonstration")
        vector_fields, case_params = create_dummy_fluid_data(
            num_cases=args.num_cases,
            num_timesteps=args.num_timesteps,
            num_params=args.num_case_params,
        )

    # Normalize data
    vector_fields, norm_stats = normalize_vector_fields(vector_fields)
    logger.info(f"Data shape: {vector_fields.shape}")
    logger.info(f"Case params shape: {case_params.shape}")

    # Create dataset
    dataset = FluidDynamicsDataset(
        vector_fields=vector_fields,
        case_params=case_params,
        sequence_length=3,
    )

    # Split into train and validation
    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create model
    model = FluidDynamicsUNet(
        in_channels=2,
        out_channels=2,
        model_channels=args.model_channels,
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
        num_case_params=args.num_case_params,
        use_fourier_conditioning=args.use_fourier_conditioning,
        num_fourier_freqs=args.num_fourier_freqs,
    ).to(device)

    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.lr * 0.01,
    )

    # Create probability path for flow matching
    path = CondOTProbPath()

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(args.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.num_epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, path, device, args, epoch)
        logger.info(f"Train loss: {train_loss:.6f}")

        # Validate
        val_loss = validate(model, val_loader, path, device, args)
        logger.info(f"Validation loss: {val_loss:.6f}")

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Learning rate: {current_lr:.6e}")

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'norm_stats': norm_stats,
                'args': vars(args),
            }
            torch.save(checkpoint, save_dir / 'best_model.pt')
            logger.info(f"Saved best model with validation loss: {val_loss:.6f}")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'norm_stats': norm_stats,
                'args': vars(args),
            }
            torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch+1}.pt')

    logger.info("\nTraining completed!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
