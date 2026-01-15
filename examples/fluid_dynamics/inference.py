"""
Inference script for fluid dynamics prediction using trained flow matching model.

This demonstrates how to:
1. Load a trained model
2. Perform single-step prediction
3. Perform multi-step autoregressive prediction
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from models.fluid_unet import FluidDynamicsUNet
from data.fluid_dataset import create_dummy_fluid_data, normalize_vector_fields, denormalize_vector_fields
from flow_matching.solver import ODESolver


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for fluid dynamics model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--num_steps", type=int, default=50,
                        help="Number of ODE integration steps")
    parser.add_argument("--num_predictions", type=int, default=10,
                        help="Number of autoregressive prediction steps")
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                        help="Classifier-free guidance scale (1.0=no guidance, >1.0=stronger conditioning)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize the predictions")
    return parser.parse_args()


class FluidPredictor:
    """Wrapper class for fluid dynamics prediction."""

    def __init__(self, model, device, norm_stats=None):
        self.model = model
        self.device = device
        self.norm_stats = norm_stats
        self.model.eval()

    @torch.no_grad()
    def predict_next_state(self, x_prev_1, x_prev_2, case_params=None, num_steps=50, guidance_scale=1.0):
        """
        Predict the next state given two previous states with optional classifier-free guidance.

        Args:
            x_prev_1: Previous state at t-1, shape (B, 2, H, W)
            x_prev_2: Previous state at t-2, shape (B, 2, H, W)
            case_params: Optional case parameters, shape (B, D)
            num_steps: Number of ODE integration steps
            guidance_scale: Classifier-free guidance scale (1.0=no guidance, >1.0=stronger)

        Returns:
            Predicted next state, shape (B, 2, H, W)
        """
        # Move to device
        x_prev_1 = x_prev_1.to(self.device)
        x_prev_2 = x_prev_2.to(self.device)
        if case_params is not None:
            case_params = case_params.to(self.device)

        # Create model wrapper for ODE solver
        class ModelWrapper(nn.Module):
            def __init__(self, model, x_prev_1, x_prev_2, case_params, guidance_scale):
                super().__init__()
                self.model = model
                self.x_prev_1 = x_prev_1
                self.x_prev_2 = x_prev_2
                self.case_params = case_params
                self.guidance_scale = guidance_scale

            def forward(self, t, x):
                # Handle both scalar and batched timesteps
                batch_size = x.shape[0]
                if t.dim() == 0:
                    t = t.repeat(batch_size)

                if self.guidance_scale == 1.0 or self.case_params is None:
                    # No guidance or no case params
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

                    # Unconditional prediction
                    extra_uncond = {
                        'x_prev_1': self.x_prev_1,
                        'x_prev_2': self.x_prev_2,
                        'case_params': None
                    }
                    v_uncond = self.model(x, t, extra_uncond)

                    # Guided velocity
                    return v_uncond + self.guidance_scale * (v_cond - v_uncond)

        wrapped_model = ModelWrapper(self.model, x_prev_1, x_prev_2, case_params, guidance_scale)

        # Create ODE solver
        solver = ODESolver(wrapped_model)

        # Start from Gaussian noise
        x_0 = torch.randn_like(x_prev_1)

        # Solve ODE from t=0 to t=1
        x_1 = solver.sample(x_0, step_size=1.0 / num_steps)

        return x_1

    @torch.no_grad()
    def predict_sequence(self, x_init_1, x_init_2, case_params=None, num_predictions=10, num_steps=50, guidance_scale=1.0):
        """
        Perform autoregressive prediction for multiple timesteps.

        Args:
            x_init_1: Initial state at t-1, shape (B, 2, H, W)
            x_init_2: Initial state at t-2, shape (B, 2, H, W)
            case_params: Optional case parameters, shape (B, D)
            num_predictions: Number of steps to predict
            num_steps: Number of ODE integration steps per prediction
            guidance_scale: Classifier-free guidance scale

        Returns:
            List of predicted states, each of shape (B, 2, H, W)
        """
        predictions = []
        x_prev_2 = x_init_2
        x_prev_1 = x_init_1

        for i in range(num_predictions):
            # Predict next state
            x_next = self.predict_next_state(x_prev_1, x_prev_2, case_params, num_steps, guidance_scale)
            predictions.append(x_next.cpu())

            # Update history
            x_prev_2 = x_prev_1
            x_prev_1 = x_next

        return predictions


def visualize_vector_field(u, v, title="Vector Field", ax=None):
    """
    Visualize a 2D vector field.

    Args:
        u: U component of velocity, shape (H, W)
        v: V component of velocity, shape (H, W)
        title: Plot title
        ax: Matplotlib axis (if None, creates new figure)
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Convert to numpy
    if torch.is_tensor(u):
        u = u.cpu().numpy()
    if torch.is_tensor(v):
        v = v.cpu().numpy()

    # Compute magnitude
    magnitude = np.sqrt(u**2 + v**2)

    # Downsample for quiver plot
    skip = 4
    x = np.arange(0, u.shape[1], skip)
    y = np.arange(0, u.shape[0], skip)
    X, Y = np.meshgrid(x, y)

    # Plot magnitude as background
    im = ax.imshow(magnitude, cmap='viridis', origin='lower')
    plt.colorbar(im, ax=ax, label='Velocity Magnitude')

    # Overlay vector field
    ax.quiver(
        X, Y,
        u[::skip, ::skip],
        v[::skip, ::skip],
        color='white',
        alpha=0.7,
        scale=10,
    )

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    return ax


def main():
    args = parse_args()

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # Get model parameters from checkpoint
    model_args = checkpoint['args']

    # Create model
    model = FluidDynamicsUNet(
        in_channels=2,
        out_channels=2,
        model_channels=model_args['model_channels'],
        num_res_blocks=model_args['num_res_blocks'],
        dropout=model_args['dropout'],
        num_case_params=model_args['num_case_params'],
        use_fourier_conditioning=model_args['use_fourier_conditioning'],
        num_fourier_freqs=model_args['num_fourier_freqs'],
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()

    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print(f"Validation loss: {checkpoint['val_loss']:.6f}")

    # Get normalization stats
    norm_stats = checkpoint.get('norm_stats', None)

    # Create predictor
    predictor = FluidPredictor(model, args.device, norm_stats)

    # Create or load test data
    print("\nGenerating test data...")
    vector_fields, case_params = create_dummy_fluid_data(
        num_cases=5,
        num_timesteps=20,
        num_params=model_args['num_case_params'],
    )

    # Normalize
    if norm_stats is not None:
        # Use training normalization stats
        mean = norm_stats['mean']
        std = norm_stats['std']
        while mean.ndim < vector_fields.ndim:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        vector_fields = (vector_fields - mean) / (std + 1e-8)

    # Take first case for visualization
    test_case_idx = 0
    x_init_2 = vector_fields[test_case_idx, 0:1]  # (1, 2, H, W)
    x_init_1 = vector_fields[test_case_idx, 1:2]  # (1, 2, H, W)
    test_case_params = case_params[test_case_idx:test_case_idx+1]  # (1, D)

    # Ground truth sequence
    ground_truth = vector_fields[test_case_idx, 2:2+args.num_predictions]

    print(f"\nRunning autoregressive prediction for {args.num_predictions} steps...")
    print(f"Using guidance scale: {args.guidance_scale}")
    predictions = predictor.predict_sequence(
        x_init_1,
        x_init_2,
        test_case_params,
        num_predictions=args.num_predictions,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
    )

    # Compute errors
    predictions_tensor = torch.stack(predictions).squeeze(1)  # (T, 2, H, W)
    mse_per_step = torch.nn.functional.mse_loss(
        predictions_tensor,
        ground_truth,
        reduction='none'
    ).mean(dim=(1, 2, 3))

    print("\nMean Squared Error per step:")
    for i, mse in enumerate(mse_per_step):
        print(f"  Step {i+1}: {mse.item():.6f}")

    avg_mse = mse_per_step.mean().item()
    print(f"\nAverage MSE: {avg_mse:.6f}")

    # Visualize if requested
    if args.visualize:
        print("\nGenerating visualizations...")

        # Select a few timesteps to visualize
        vis_steps = min(6, args.num_predictions)
        step_indices = np.linspace(0, args.num_predictions-1, vis_steps, dtype=int)

        fig, axes = plt.subplots(3, vis_steps, figsize=(4*vis_steps, 12))

        for i, step_idx in enumerate(step_indices):
            # Ground truth
            gt_u = ground_truth[step_idx, 0]
            gt_v = ground_truth[step_idx, 1]
            visualize_vector_field(gt_u, gt_v, f"Ground Truth t={step_idx+3}", ax=axes[0, i])

            # Prediction
            pred_u = predictions[step_idx][0, 0]
            pred_v = predictions[step_idx][0, 1]
            visualize_vector_field(pred_u, pred_v, f"Prediction t={step_idx+3}", ax=axes[1, i])

            # Error
            error_u = torch.abs(pred_u - gt_u)
            error_v = torch.abs(pred_v - gt_v)
            error_mag = torch.sqrt(error_u**2 + error_v**2)

            im = axes[2, i].imshow(error_mag.cpu().numpy(), cmap='hot', origin='lower')
            plt.colorbar(im, ax=axes[2, i], label='Error Magnitude')
            axes[2, i].set_title(f"Error t={step_idx+3}")
            axes[2, i].set_xlabel('X')
            axes[2, i].set_ylabel('Y')

        plt.tight_layout()
        plt.savefig('fluid_predictions.png', dpi=150, bbox_inches='tight')
        print("Saved visualization to fluid_predictions.png")
        plt.show()


if __name__ == "__main__":
    main()
