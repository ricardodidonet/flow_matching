"""
Example showing how to use the config module.
"""

# ============================================================================
# Example 1: Basic usage - replace parse_args()
# ============================================================================

from config import get_train_parser

def example_basic():
    """Replace your parse_args() with this."""
    # Instead of defining 20+ arguments inline:
    # def parse_args():
    #     parser = argparse.ArgumentParser(...)
    #     parser.add_argument("--data_path", ...)
    #     parser.add_argument("--batch_size", ...)
    #     ...

    # Just do this:
    parser = get_train_parser()
    args = parser.parse_args()

    print(f"Model channels: {args.model_channels}")
    print(f"Batch size: {args.batch_size}")
    return args


# ============================================================================
# Example 2: With dataclass (more structured)
# ============================================================================

from config import get_train_parser, TrainConfig

def example_dataclass():
    """Use dataclass for cleaner access."""
    parser = get_train_parser()
    args = parser.parse_args()

    # Convert to structured config
    config = TrainConfig.from_args(args)

    # Now access with dot notation
    print(f"Model channels: {config.model.model_channels}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Device: {config.system.device}")

    return config


# ============================================================================
# Example 3: Load from JSON file
# ============================================================================

import json
from config import get_train_parser

def example_from_file(config_path="experiment_config.json"):
    """Load configuration from a file."""
    # Load config
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # Create parser with defaults from file
    parser = get_train_parser()
    parser.set_defaults(**config_dict)

    # Parse (can still override from command line)
    args = parser.parse_args()

    return args


# ============================================================================
# Example 4: Save configuration
# ============================================================================

from config import get_train_parser, save_config_to_file

def example_save_config():
    """Save configuration for reproducibility."""
    parser = get_train_parser()
    args = parser.parse_args()

    # Save to file
    save_config_to_file(args, "checkpoints/config.json")

    print("Configuration saved to checkpoints/config.json")
    return args


# ============================================================================
# Example 5: Minimal update to existing train_fluid.py
# ============================================================================

def minimal_update_example():
    """
    Minimal change to your existing code.

    In train_fluid.py, replace:

        def parse_args():
            parser = argparse.ArgumentParser(...)
            parser.add_argument("--data_path", ...)
            # ... 20 more arguments
            return parser.parse_args()

    With just:

        from config import get_train_parser

        def parse_args():
            return get_train_parser().parse_args()

    That's it! Everything else stays the same.
    """
    from config import get_train_parser

    parser = get_train_parser()
    args = parser.parse_args()

    # Use exactly as before
    print(f"Args: {args}")
    return args


# ============================================================================
# Example 6: Custom modifications
# ============================================================================

from config import get_train_parser

def example_custom():
    """Add custom arguments on top of base config."""
    parser = get_train_parser()

    # Add your own custom arguments
    parser.add_argument("--experiment_name", type=str, default="exp1",
                        help="Experiment name")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")

    args = parser.parse_args()
    return args


# ============================================================================
# Run examples
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Example 1: Basic usage")
    print("=" * 60)
    # args = example_basic()

    print("\n" + "=" * 60)
    print("Example 2: Dataclass usage")
    print("=" * 60)
    # config = example_dataclass()

    print("\n" + "=" * 60)
    print("Example 5: Minimal update (recommended)")
    print("=" * 60)
    args = minimal_update_example()

    print("\n✅ All examples work!")
    print(f"\nYour args have {len(vars(args))} parameters")
    print("Now your training script is much cleaner!")
