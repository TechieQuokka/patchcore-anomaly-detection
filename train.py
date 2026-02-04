"""
PatchCore Training Script

Builds memory bank for anomaly detection by:
1. Extracting features from normal training images
2. Applying Greedy Coreset subsampling
3. Building faiss index for fast k-NN search
"""

import os
import argparse
from pathlib import Path
import time

import torch

from config import get_config, Config
from patchcore import PatchCore
from dataset import get_dataloader, MVTEC_CATEGORIES


def train_category(
    category: str,
    config: Config = None
) -> PatchCore:
    """
    Train PatchCore on a single MVTec AD category.

    Args:
        category: Category name
        config: Config instance (uses default if None)

    Returns:
        Trained PatchCore model
    """
    if config is None:
        config = get_config()

    print(f"\n{'='*60}")
    print(f"Training PatchCore on: {category}")
    print(f"{'='*60}")

    # Create output directory
    output_path = Path(config.train.checkpoint_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create dataloader (train split, normal samples only)
    print(f"\nLoading training data...")
    train_loader = get_dataloader(
        root=config.data.root_path,
        category=category,
        split='train',
        batch_size=config.train.batch_size,
        num_workers=config.data.num_workers,
        shuffle=False  # No need to shuffle for feature extraction
    )
    print(f"  Training samples: {len(train_loader.dataset)}")

    # Initialize PatchCore
    print(f"\nInitializing PatchCore...")
    print(f"  Device: {config.train.device}")
    print(f"  Backbone: {config.model.backbone}")
    print(f"  Coreset ratio: {config.model.coreset_ratio}")
    print(f"  Batch size: {config.train.batch_size}")

    model = PatchCore(
        backbone=config.model.backbone,
        device=config.train.device,
        config=config.model
    )

    # Build memory bank
    start_time = time.time()
    model.fit(train_loader)
    elapsed = time.time() - start_time

    print(f"\nTraining completed in {elapsed:.1f} seconds")

    # Save memory bank
    memory_bank_path = output_path / f"{category}_memory_bank.npy"
    model.save(str(memory_bank_path))

    return model


def train_all_categories(config: Config = None) -> None:
    """
    Train PatchCore on all MVTec AD categories.

    Args:
        config: Config instance (uses default if None)
    """
    if config is None:
        config = get_config()

    categories = config.data.categories

    print(f"\n{'#'*60}")
    print(f"Training PatchCore on ALL {len(categories)} categories")
    print(f"{'#'*60}")

    total_start = time.time()

    for i, category in enumerate(categories, 1):
        print(f"\n[{i}/{len(categories)}]", end="")
        train_category(category, config)

    total_elapsed = time.time() - total_start
    print(f"\n{'#'*60}")
    print(f"All training completed in {total_elapsed/60:.1f} minutes")
    print(f"Memory banks saved to: {config.train.checkpoint_dir}")
    print(f"{'#'*60}")


def main():
    # Get default config
    cfg = get_config()

    parser = argparse.ArgumentParser(
        description='Train PatchCore on MVTec AD dataset'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default=cfg.data.root_path,
        help='Path to MVTec AD dataset'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=cfg.train.checkpoint_dir,
        help='Directory to save memory banks'
    )
    parser.add_argument(
        '--category',
        type=str,
        default='all',
        choices=['all'] + MVTEC_CATEGORIES,
        help='Category to train (default: all)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=cfg.train.batch_size,
        help='Batch size for feature extraction'
    )
    parser.add_argument(
        '--coreset_ratio',
        type=float,
        default=cfg.model.coreset_ratio,
        help='Ratio for coreset subsampling'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=cfg.train.device,
        choices=['cuda', 'cpu'],
        help='Device to use'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=cfg.data.num_workers,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='Skip categories that already have memory banks'
    )

    args = parser.parse_args()

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    if args.device == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Update config with CLI arguments
    cfg.data.root_path = args.data_path
    cfg.train.checkpoint_dir = args.output_dir
    cfg.train.batch_size = args.batch_size
    cfg.train.device = args.device
    cfg.model.coreset_ratio = args.coreset_ratio
    cfg.data.num_workers = args.num_workers

    # Train
    if args.category == 'all':
        for i, category in enumerate(cfg.data.categories, 1):
            memory_bank_path = Path(cfg.train.checkpoint_dir) / f"{category}_memory_bank.npy"
            if args.skip_existing and memory_bank_path.exists():
                print(f"[{i}/{len(cfg.data.categories)}] Skipping {category} (already exists)")
                continue
            print(f"\n[{i}/{len(cfg.data.categories)}]", end="")
            train_category(category, cfg)
    else:
        memory_bank_path = Path(cfg.train.checkpoint_dir) / f"{args.category}_memory_bank.npy"
        if args.skip_existing and memory_bank_path.exists():
            print(f"Skipping {args.category} (already exists)")
        else:
            train_category(args.category, cfg)


if __name__ == '__main__':
    main()
