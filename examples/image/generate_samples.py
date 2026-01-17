#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

"""
Standalone script to generate images from a trained flow matching model.

Usage:
    # Generate 16 random class images
    python generate_samples.py --checkpoint output_dir/checkpoint-100.pth --num_samples 16

    # Generate 64 dogs (class 5) with CFG
    python generate_samples.py --checkpoint output_dir/checkpoint-100.pth --num_samples 64 --label 5 --cfg_scale 0.5

    # Generate grid of all 10 CIFAR-10 classes
    python generate_samples.py --checkpoint output_dir/checkpoint-100.pth --num_samples 80 --all_classes --cfg_scale 0.3
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import numpy as np
from torchvision.utils import save_image
from models.model_configs import instantiate_model
from flow_matching.solver.ode_solver import ODESolver
from flow_matching.utils import ModelWrapper
from models.ema import EMA
from training.edm_time_discretization import get_time_discretization

logger = logging.getLogger(__name__)

# CIFAR-10 class names
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


class CFGScaledModel(ModelWrapper):
    """Wrapper for classifier-free guidance during sampling."""

    def __init__(self, model):
        super().__init__(model)
        self.nfe_counter = 0

    def forward(self, x, t, cfg_scale, label):
        t = torch.zeros(x.shape[0], device=x.device) + t

        if cfg_scale != 0.0:
            # Generate with and without conditioning
            with torch.cuda.amp.autocast(), torch.no_grad():
                conditional = self.model(x, t, extra={"label": label})
                condition_free = self.model(x, t, extra={})
            # CFG: extrapolate in conditional direction
            result = (1.0 + cfg_scale) * conditional - cfg_scale * condition_free
        else:
            # No guidance
            with torch.cuda.amp.autocast(), torch.no_grad():
                result = self.model(x, t, extra={"label": label})

        self.nfe_counter += 1
        return result.to(dtype=torch.float32)

    def reset_nfe_counter(self):
        self.nfe_counter = 0

    def get_nfe(self):
        return self.nfe_counter


def load_checkpoint(checkpoint_path, dataset, use_ema, device):
    """Load model from checkpoint file."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # Create model
    model = instantiate_model(
        architechture=dataset,
        is_discrete=False,
        use_ema=use_ema,
    )

    # Load checkpoint
    # Note: weights_only=False is safe here since this is your own checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Handle EMA models
    if use_ema and isinstance(model, EMA):
        # Load EMA weights
        if 'model_ema' in checkpoint:
            model.load_state_dict(checkpoint['model_ema'])
            logger.info("Loaded EMA weights")
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            logger.info("Loaded model weights (EMA weights not found)")
        else:
            raise ValueError("No model weights found in checkpoint")
    else:
        # Load regular model weights
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            logger.info("Loaded model weights")
        else:
            raise ValueError("No model weights found in checkpoint")

    model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', 'unknown')
    logger.info(f"Loaded checkpoint from epoch {epoch}")

    return model


def generate_samples(
    model,
    num_samples,
    labels,
    cfg_scale,
    ode_method,
    ode_options,
    edm_schedule,
    device,
):
    """Generate samples using the trained model."""

    # Wrap model with CFG
    cfg_model = CFGScaledModel(model)
    cfg_model.train(False)

    # Create ODE solver
    solver = ODESolver(velocity_model=cfg_model)

    # Generate initial noise
    logger.info(f"Generating {num_samples} samples...")
    x_0 = torch.randn(num_samples, 3, 32, 32, dtype=torch.float32, device=device)

    # Prepare labels
    if labels is None:
        labels = torch.randint(0, 10, (num_samples,), device=device)
        logger.info(f"Using random labels")
    else:
        labels = torch.tensor(labels, dtype=torch.long, device=device)
        logger.info(f"Using specified labels: {labels.tolist()[:10]}{'...' if len(labels) > 10 else ''}")

    # Create time grid
    if edm_schedule:
        time_grid = get_time_discretization(nfes=ode_options.get("nfe", 50))
    else:
        time_grid = torch.tensor([0.0, 1.0], device=device)

    logger.info(f"ODE method: {ode_method}, cfg_scale: {cfg_scale}")

    # Sample from the model
    cfg_model.reset_nfe_counter()
    synthetic_samples = solver.sample(
        time_grid=time_grid,
        x_init=x_0,
        method=ode_method,
        return_intermediates=False,
        atol=ode_options.get("atol", 1e-5),
        rtol=ode_options.get("rtol", 1e-5),
        step_size=ode_options.get("step_size"),
        label=labels,
        cfg_scale=cfg_scale,
    )

    # Post-process: scale from [-1, 1] to [0, 1]
    synthetic_samples = torch.clamp(
        synthetic_samples * 0.5 + 0.5, min=0.0, max=1.0
    )
    # Quantize to 8-bit for consistency
    synthetic_samples = torch.floor(synthetic_samples * 255)
    synthetic_samples = synthetic_samples.to(torch.float32) / 255.0

    nfe = cfg_model.get_nfe()
    logger.info(f"Generated {num_samples} samples in {nfe} function evaluations")
    logger.info(f"Average NFE per sample: {nfe / num_samples:.1f}")

    return synthetic_samples, labels


def save_samples(samples, labels, output_dir, save_individual, nrow):
    """Save generated samples as grid and/or individual images."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save grid
    grid_path = output_dir / "samples_grid.png"
    save_image(samples, grid_path, nrow=nrow, padding=2)
    logger.info(f"Saved grid to {grid_path}")

    # Save individual images
    if save_individual:
        individual_dir = output_dir / "individual"
        individual_dir.mkdir(exist_ok=True)

        for i, (img, label) in enumerate(zip(samples, labels)):
            class_name = CIFAR10_CLASSES[label.item()]
            img_path = individual_dir / f"sample_{i:04d}_class{label.item()}_{class_name}.png"
            save_image(img, img_path)

        logger.info(f"Saved {len(samples)} individual images to {individual_dir}")

    # Save class distribution summary
    summary_path = output_dir / "generation_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Total samples: {len(samples)}\n")
        f.write(f"Classes generated:\n")
        for class_idx in range(10):
            count = (labels == class_idx).sum().item()
            if count > 0:
                f.write(f"  {class_idx} ({CIFAR10_CLASSES[class_idx]}): {count} samples\n")

    logger.info(f"Saved summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate images from trained flow matching model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model loading
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (.pth)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "imagenet"],
        help="Dataset the model was trained on"
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use EMA weights for generation (better quality if model was trained with --use_ema)"
    )

    # Generation parameters
    parser.add_argument(
        "--num_samples",
        type=int,
        default=64,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--label",
        type=int,
        default=None,
        help="Class label for all samples (0-9 for CIFAR-10). If not specified, random labels are used."
    )
    parser.add_argument(
        "--all_classes",
        action="store_true",
        help="Generate equal number of samples for each class. Overrides --label."
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=0.0,
        help="Classifier-free guidance scale. 0.0=no guidance, 0.3-0.5=typical, 1.0+=strong"
    )

    # ODE solver parameters
    parser.add_argument(
        "--ode_method",
        type=str,
        default="midpoint",
        choices=["euler", "midpoint", "heun2", "dopri5"],
        help="ODE solver method"
    )
    parser.add_argument(
        "--step_size",
        type=float,
        default=0.05,
        help="Step size for fixed-step ODE solvers (euler, midpoint, heun2)"
    )
    parser.add_argument(
        "--nfe",
        type=int,
        default=50,
        help="Number of function evaluations for adaptive solvers (dopri5) or EDM schedule"
    )
    parser.add_argument(
        "--edm_schedule",
        action="store_true",
        help="Use EDM time discretization schedule"
    )

    # Output parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_samples",
        help="Directory to save generated images"
    )
    parser.add_argument(
        "--save_individual",
        action="store_true",
        help="Save individual images in addition to grid"
    )
    parser.add_argument(
        "--nrow",
        type=int,
        default=8,
        help="Number of images per row in grid"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load model
    model = load_checkpoint(
        args.checkpoint,
        args.dataset,
        args.use_ema,
        device
    )

    # Prepare labels
    if args.all_classes:
        # Generate equal samples for each class
        samples_per_class = args.num_samples // 10
        remainder = args.num_samples % 10
        labels = []
        for class_idx in range(10):
            count = samples_per_class + (1 if class_idx < remainder else 0)
            labels.extend([class_idx] * count)
        logger.info(f"Generating {samples_per_class} samples per class (+ {remainder} extra)")
    elif args.label is not None:
        # Use specified label for all samples
        labels = [args.label] * args.num_samples
        logger.info(f"Generating {args.num_samples} samples of class {args.label} ({CIFAR10_CLASSES[args.label]})")
    else:
        # Random labels (will be assigned in generate_samples)
        labels = None

    # ODE options
    ode_options = {
        "step_size": args.step_size,
        "nfe": args.nfe,
        "atol": 1e-5,
        "rtol": 1e-5,
    }

    # Generate samples
    samples, labels = generate_samples(
        model=model,
        num_samples=args.num_samples,
        labels=labels,
        cfg_scale=args.cfg_scale,
        ode_method=args.ode_method,
        ode_options=ode_options,
        edm_schedule=args.edm_schedule,
        device=device,
    )

    # Save samples
    save_samples(
        samples=samples,
        labels=labels,
        output_dir=args.output_dir,
        save_individual=args.save_individual,
        nrow=args.nrow,
    )

    logger.info("Done!")

    # Print example usage for viewing
    logger.info(f"\nView the generated grid: {Path(args.output_dir) / 'samples_grid.png'}")
    if args.save_individual:
        logger.info(f"Individual images: {Path(args.output_dir) / 'individual/'}")


if __name__ == "__main__":
    main()
