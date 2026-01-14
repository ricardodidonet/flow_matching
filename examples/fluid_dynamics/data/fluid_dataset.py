"""
Dataset for fluid dynamics data with autoregressive structure.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, Tuple


class FluidDynamicsDataset(Dataset):
    """
    Dataset for fluid dynamics vector field data.

    Expected data format:
    - Vector fields: (N, T, 2, H, W) where
        N = number of simulations/cases
        T = number of timesteps
        2 = velocity components (u, v)
        H, W = spatial dimensions (64, 64)

    - Case parameters (optional): (N, D) where
        D = number of case parameters (e.g., Reynolds number, cylinder radius, etc.)
    """

    def __init__(
        self,
        vector_fields: torch.Tensor,
        case_params: Optional[torch.Tensor] = None,
        sequence_length: int = 3,
    ):
        """
        Args:
            vector_fields: Tensor of shape (N, T, 2, H, W)
            case_params: Optional tensor of shape (N, D) with case parameters
            sequence_length: Number of timesteps to use (default: 3 for t-2, t-1, t)
        """
        super().__init__()

        assert vector_fields.ndim == 5, "vector_fields must be 5D: (N, T, 2, H, W)"
        assert sequence_length >= 3, "Need at least 3 timesteps for t-2, t-1, t"

        self.vector_fields = vector_fields
        self.case_params = case_params
        self.sequence_length = sequence_length

        self.num_cases = vector_fields.shape[0]
        self.num_timesteps = vector_fields.shape[1]

        # Validate shapes
        assert self.num_timesteps >= sequence_length, \
            f"Need at least {sequence_length} timesteps, got {self.num_timesteps}"

        if case_params is not None:
            assert case_params.shape[0] == self.num_cases, \
                "case_params must have same number of cases as vector_fields"

    def __len__(self):
        # For each case, we can create (T - sequence_length + 1) samples
        return self.num_cases * (self.num_timesteps - self.sequence_length + 1)

    def __getitem__(self, idx):
        # Determine which case and which time window
        case_idx = idx // (self.num_timesteps - self.sequence_length + 1)
        time_idx = idx % (self.num_timesteps - self.sequence_length + 1)

        # Get the sequence: [time_idx, time_idx+1, time_idx+2]
        # This corresponds to [t-2, t-1, t]
        x_t_minus_2 = self.vector_fields[case_idx, time_idx]
        x_t_minus_1 = self.vector_fields[case_idx, time_idx + 1]
        x_t = self.vector_fields[case_idx, time_idx + 2]

        sample = {
            'x_target': x_t,  # Target state at time t
            'x_prev_1': x_t_minus_1,  # Previous state at t-1
            'x_prev_2': x_t_minus_2,  # Previous state at t-2
        }

        if self.case_params is not None:
            sample['case_params'] = self.case_params[case_idx]

        return sample


def create_dummy_fluid_data(
    num_cases: int = 100,
    num_timesteps: int = 50,
    height: int = 64,
    width: int = 64,
    num_params: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create dummy fluid dynamics data for testing.

    This generates synthetic vector field data that simulates fluid flow patterns.

    Args:
        num_cases: Number of simulation cases
        num_timesteps: Number of timesteps per case
        height: Height of the spatial grid
        width: Width of the spatial grid
        num_params: Number of case parameters

    Returns:
        vector_fields: (num_cases, num_timesteps, 2, height, width)
        case_params: (num_cases, num_params)
    """
    # Create spatially correlated velocity fields
    vector_fields = torch.zeros(num_cases, num_timesteps, 2, height, width)

    for case in range(num_cases):
        # Random parameters for this case
        amplitude = 0.5 + torch.rand(1) * 0.5
        frequency = 0.05 + torch.rand(1) * 0.1
        phase = torch.rand(1) * 2 * np.pi

        # Create grid
        x = torch.linspace(0, 2 * np.pi, width)
        y = torch.linspace(0, 2 * np.pi, height)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        for t in range(num_timesteps):
            # Create time-evolving velocity field
            time_factor = t / num_timesteps

            # U component (horizontal velocity)
            u = amplitude * torch.sin(frequency * X + phase) * torch.cos(frequency * Y)
            u = u * (1 + 0.2 * torch.sin(2 * np.pi * time_factor))

            # V component (vertical velocity)
            v = amplitude * torch.cos(frequency * X + phase) * torch.sin(frequency * Y)
            v = v * (1 + 0.2 * torch.cos(2 * np.pi * time_factor))

            vector_fields[case, t, 0] = u
            vector_fields[case, t, 1] = v

    # Create random case parameters
    # For example: Reynolds number, cylinder radius, inlet velocity
    case_params = torch.rand(num_cases, num_params)

    # Normalize to reasonable ranges
    # Param 0: Reynolds number (scaled to [100, 10000])
    case_params[:, 0] = case_params[:, 0] * 9900 + 100

    # Param 1: Cylinder radius (scaled to [0.1, 0.5])
    if num_params > 1:
        case_params[:, 1] = case_params[:, 1] * 0.4 + 0.1

    # Param 2: Inlet velocity (scaled to [0.5, 2.0])
    if num_params > 2:
        case_params[:, 2] = case_params[:, 2] * 1.5 + 0.5

    return vector_fields, case_params


def normalize_vector_fields(vector_fields: torch.Tensor) -> Tuple[torch.Tensor, dict]:
    """
    Normalize vector fields to have zero mean and unit variance.

    Args:
        vector_fields: (N, T, 2, H, W)

    Returns:
        Normalized vector fields and normalization stats
    """
    # Compute statistics across all dimensions except channel
    mean = vector_fields.mean(dim=(0, 1, 3, 4), keepdim=True)
    std = vector_fields.std(dim=(0, 1, 3, 4), keepdim=True)

    normalized = (vector_fields - mean) / (std + 1e-8)

    stats = {
        'mean': mean.squeeze(),
        'std': std.squeeze(),
    }

    return normalized, stats


def denormalize_vector_fields(
    normalized_fields: torch.Tensor,
    stats: dict
) -> torch.Tensor:
    """
    Denormalize vector fields back to original scale.

    Args:
        normalized_fields: Normalized vector fields
        stats: Dictionary with 'mean' and 'std'

    Returns:
        Denormalized vector fields
    """
    mean = stats['mean']
    std = stats['std']

    # Broadcast to match dimensions
    while mean.ndim < normalized_fields.ndim:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    return normalized_fields * (std + 1e-8) + mean
