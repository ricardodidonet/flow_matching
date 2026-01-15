"""
Training script for fluid dynamics prediction using flow matching.

Now with cleaner configuration management!
"""

import logging
import sys
from pathlib import Path
import torch

# Import config
from config import get_train_parser, save_config_to_file

# Import training utilities
from models.fluid_unet import FluidDynamicsUNet
from data.fluid_dataset import (
    FluidDynamicsDataset,
    create_dummy_fluid_data,
    normalize_vector_fields,
)
from flow_matching.path import CondOTProbPath

# Original training functions (import or copy)
from train_fluid import train_epoch, validate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def main():
    # Get parser from config module
    parser = get_train_parser()
    args = parser.parse_args()

    # Save configuration for reproducibility
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_config_to_file(args, save_dir / "config.json")

    # Set up device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load or create data
    if args.data_path is not None:
        logger.info(f"Loading data from {args.data_path}")
        raise NotImplementedError("Custom data loading not yet implemented")
    else:
        logger.info("Creating dummy data")
        vector_fields, case_params = create_dummy_fluid_data(
            num_cases=args.num_cases,
            num_timesteps=args.num_timesteps,
            num_params=args.num_case_params,
        )

    # Normalize
    vector_fields, norm_stats = normalize_vector_fields(vector_fields)
    logger.info(f"Data shape: {vector_fields.shape}")

    # Create dataset and dataloaders
    from torch.utils.data import DataLoader, random_split

    dataset = FluidDynamicsDataset(
        vector_fields=vector_fields,
        case_params=case_params,
        sequence_length=3,
    )

    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

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

    logger.info(f"Model: {sum(p.numel() for p in model.parameters())} parameters")

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.lr * 0.01,
    )

    # Create probability path
    path = CondOTProbPath()

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(args.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.num_epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, path, device, args, epoch)
        logger.info(f"Train loss: {train_loss:.6f}")

        val_loss = validate(model, val_loader, path, device, args)
        logger.info(f"Validation loss: {val_loss:.6f}")

        scheduler.step()

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
                'args': vars(args),  # Save all arguments
            }
            torch.save(checkpoint, save_dir / 'best_model.pt')
            logger.info(f"Saved best model with validation loss: {val_loss:.6f}")

    logger.info("\nTraining completed!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
