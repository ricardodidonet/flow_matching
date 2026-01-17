"""
Script to run evaluation on trained fluid dynamics model.

Usage:
    python run_eval.py --checkpoint checkpoints/best_model.pt --cfg_scale 1.5
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.fluid_unet import FluidDynamicsUNet
from data.fluid_dataset import FluidDynamicsDataset, create_dummy_fluid_data, normalize_vector_fields
from eval_fluid import eval_fluid_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fluid dynamics model")

    # Model checkpoint
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")

    # Evaluation parameters
    parser.add_argument("--cfg_scale", type=float, default=0.0,
                        help="Classifier-free guidance scale (0.0=no guidance)")
    parser.add_argument("--num_ode_steps", type=int, default=50,
                        help="Number of ODE integration steps")
    parser.add_argument("--num_rollout_steps", type=int, default=10,
                        help="Number of autoregressive rollout steps")

    # Data parameters
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")

    # Output
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Directory to save evaluation results")

    # System
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("Fluid Dynamics Model Evaluation")
    logger.info("="*60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Device: {device}")
    logger.info(f"CFG Scale: {args.cfg_scale}")
    logger.info(f"ODE Steps: {args.num_ode_steps}")
    logger.info("="*60)

    # Load checkpoint
    logger.info(f"\nLoading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Get saved arguments
    saved_args = checkpoint.get('args', {})
    logger.info(f"Model was trained with:")
    logger.info(f"  Model channels: {saved_args.get('model_channels', 128)}")
    logger.info(f"  Dropout: {saved_args.get('dropout', 0.1)}")
    logger.info(f"  Num case params: {saved_args.get('num_case_params', 3)}")

    # Create model with saved configuration
    model = FluidDynamicsUNet(
        in_channels=2,
        out_channels=2,
        model_channels=saved_args.get('model_channels', 128),
        num_res_blocks=saved_args.get('num_res_blocks', 2),
        dropout=saved_args.get('dropout', 0.1),
        num_case_params=saved_args.get('num_case_params', 3),
        use_fourier_conditioning=saved_args.get('use_fourier_conditioning', True),
        num_fourier_freqs=saved_args.get('num_fourier_freqs', 16),
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")

    # Load or create validation data
    logger.info("\nPreparing validation data...")

    # For demonstration, create dummy data
    # TODO: Replace with your actual validation data
    vector_fields, case_params = create_dummy_fluid_data(
        num_cases=20,
        num_timesteps=15,
        num_params=saved_args.get('num_case_params', 3),
    )

    # Normalize with saved stats if available
    norm_stats = checkpoint.get('norm_stats', None)
    if norm_stats is not None:
        mean = norm_stats['mean']
        std = norm_stats['std']
        while mean.ndim < vector_fields.ndim:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        vector_fields = (vector_fields - mean) / (std + 1e-8)
    else:
        vector_fields, _ = normalize_vector_fields(vector_fields)

    # Create dataset
    dataset = FluidDynamicsDataset(
        vector_fields=vector_fields,
        case_params=case_params,
        sequence_length=3,
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Don't shuffle for evaluation
        num_workers=args.num_workers,
        pin_memory=True,
    )

    logger.info(f"Validation dataset: {len(dataset)} samples")

    # Run evaluation
    logger.info("\nStarting evaluation...")

    eval_metrics = eval_fluid_model(
        model=model,
        data_loader=dataloader,
        device=device,
        epoch=checkpoint.get('epoch', 0),
        cfg_scale=args.cfg_scale,
        num_ode_steps=args.num_ode_steps,
        num_rollout_steps=args.num_rollout_steps,
        output_dir=output_dir,
    )

    # Save results
    results_file = output_dir / "eval_results.txt"
    with open(results_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"Evaluation Results\n")
        f.write("="*60 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"CFG Scale: {args.cfg_scale}\n")
        f.write(f"ODE Steps: {args.num_ode_steps}\n")
        f.write("="*60 + "\n\n")

        for key, value in eval_metrics.items():
            f.write(f"{key}: {value:.6f}\n")

    logger.info(f"\nResults saved to {results_file}")
    logger.info(f"Visualizations saved to {output_dir}/snapshots/")

    # Optional: Try multiple CFG scales
    if args.cfg_scale == 0.0:
        logger.info("\n" + "="*60)
        logger.info("Tip: Try different CFG scales to see their effect:")
        logger.info("  python run_eval.py --checkpoint {} --cfg_scale 0.5".format(args.checkpoint))
        logger.info("  python run_eval.py --checkpoint {} --cfg_scale 1.0".format(args.checkpoint))
        logger.info("  python run_eval.py --checkpoint {} --cfg_scale 1.5".format(args.checkpoint))
        logger.info("="*60)


if __name__ == "__main__":
    main()
