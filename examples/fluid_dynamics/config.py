"""
Configuration management for fluid dynamics training.

This module provides argument parsers and configuration utilities.
"""

import argparse
import torch
from dataclasses import dataclass, field
from typing import Optional


def add_data_args(parser):
    """Add data-related arguments."""
    group = parser.add_argument_group('Data Parameters')
    group.add_argument("--data_path", type=str, default=None,
                       help="Path to fluid dynamics data (if None, uses dummy data)")
    group.add_argument("--num_cases", type=int, default=100,
                       help="Number of simulation cases (for dummy data)")
    group.add_argument("--num_timesteps", type=int, default=50,
                       help="Number of timesteps per case (for dummy data)")
    group.add_argument("--num_case_params", type=int, default=3,
                       help="Number of case parameters")
    return parser


def add_model_args(parser):
    """Add model architecture arguments."""
    group = parser.add_argument_group('Model Parameters')
    group.add_argument("--model_channels", type=int, default=128,
                       help="Base channel count for UNet")
    group.add_argument("--num_res_blocks", type=int, default=2,
                       help="Number of residual blocks per resolution")
    group.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout probability")
    group.add_argument("--use_fourier_conditioning", action="store_true", default=True,
                       help="Use Fourier features for case parameters")
    group.add_argument("--num_fourier_freqs", type=int, default=16,
                       help="Number of Fourier frequencies")
    return parser


def add_training_args(parser):
    """Add training-related arguments."""
    group = parser.add_argument_group('Training Parameters')
    group.add_argument("--batch_size", type=int, default=16,
                       help="Batch size")
    group.add_argument("--num_epochs", type=int, default=100,
                       help="Number of training epochs")
    group.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    group.add_argument("--weight_decay", type=float, default=1e-5,
                       help="Weight decay")
    group.add_argument("--train_split", type=float, default=0.8,
                       help="Fraction of data for training")
    return parser


def add_flow_matching_args(parser):
    """Add flow matching specific arguments."""
    group = parser.add_argument_group('Flow Matching Parameters')
    group.add_argument("--use_skewed_timesteps", action="store_true",
                       help="Use skewed timestep sampling (focus on difficult regions)")
    group.add_argument("--conditioning_drop_prob", type=float, default=0.1,
                       help="Probability of dropping conditioning for classifier-free guidance")
    return parser


def add_system_args(parser):
    """Add system/hardware arguments."""
    group = parser.add_argument_group('System Parameters')
    group.add_argument("--device", type=str,
                       default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use for training")
    group.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    group.add_argument("--save_dir", type=str, default="./checkpoints",
                       help="Directory to save checkpoints")
    group.add_argument("--log_interval", type=int, default=10,
                       help="Logging interval in batches")
    return parser


def get_train_parser():
    """Get the complete training argument parser."""
    parser = argparse.ArgumentParser(
        description="Train fluid dynamics flow matching model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Shows defaults in help
    )

    # Add all argument groups
    parser = add_data_args(parser)
    parser = add_model_args(parser)
    parser = add_training_args(parser)
    parser = add_flow_matching_args(parser)
    parser = add_system_args(parser)

    return parser


def get_inference_parser():
    """Get the inference argument parser."""
    parser = argparse.ArgumentParser(
        description="Inference for fluid dynamics model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--num_steps", type=int, default=50,
                        help="Number of ODE integration steps")
    parser.add_argument("--num_predictions", type=int, default=10,
                        help="Number of autoregressive prediction steps")
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                        help="Classifier-free guidance scale (1.0=no guidance, >1.0=stronger)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize the predictions")

    return parser


# ============================================================================
# Dataclass-based configuration (Alternative approach)
# ============================================================================

@dataclass
class DataConfig:
    """Data configuration."""
    data_path: Optional[str] = None
    num_cases: int = 100
    num_timesteps: int = 50
    num_case_params: int = 3


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_channels: int = 128
    num_res_blocks: int = 2
    dropout: float = 0.1
    use_fourier_conditioning: bool = True
    num_fourier_freqs: int = 16


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 16
    num_epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-5
    train_split: float = 0.8


@dataclass
class FlowMatchingConfig:
    """Flow matching configuration."""
    use_skewed_timesteps: bool = False
    conditioning_drop_prob: float = 0.1


@dataclass
class SystemConfig:
    """System configuration."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    save_dir: str = "./checkpoints"
    log_interval: int = 10


@dataclass
class TrainConfig:
    """Complete training configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    flow_matching: FlowMatchingConfig = field(default_factory=FlowMatchingConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    @classmethod
    def from_args(cls, args):
        """Create config from argparse Namespace."""
        return cls(
            data=DataConfig(
                data_path=args.data_path,
                num_cases=args.num_cases,
                num_timesteps=args.num_timesteps,
                num_case_params=args.num_case_params,
            ),
            model=ModelConfig(
                model_channels=args.model_channels,
                num_res_blocks=args.num_res_blocks,
                dropout=args.dropout,
                use_fourier_conditioning=args.use_fourier_conditioning,
                num_fourier_freqs=args.num_fourier_freqs,
            ),
            training=TrainingConfig(
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                train_split=args.train_split,
            ),
            flow_matching=FlowMatchingConfig(
                use_skewed_timesteps=args.use_skewed_timesteps,
                conditioning_drop_prob=args.conditioning_drop_prob,
            ),
            system=SystemConfig(
                device=args.device,
                num_workers=args.num_workers,
                save_dir=args.save_dir,
                log_interval=args.log_interval,
            ),
        )


def args_to_dict(args):
    """Convert argparse Namespace to dict."""
    return vars(args)


def load_config_from_file(config_path):
    """Load configuration from a JSON or YAML file."""
    import json
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return config_dict


def save_config_to_file(args, config_path):
    """Save configuration to a JSON file."""
    import json
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
