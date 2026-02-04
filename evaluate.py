"""
PatchCore Evaluation Script

Evaluates trained PatchCore model on MVTec AD test set:
1. Image-level AUROC
2. Pixel-level AUROC
3. Per-category results
4. Anomaly map generation
"""

import os
import argparse
from pathlib import Path
import json
import time

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from config import get_config, Config
from patchcore import PatchCore
from dataset import get_dataloader, MVTEC_CATEGORIES


def evaluate_category(
    category: str,
    config: Config = None
) -> dict:
    """
    Evaluate PatchCore on a single MVTec AD category.

    Args:
        category: Category name
        config: Config instance (uses default if None)

    Returns:
        Dictionary with evaluation metrics
    """
    if config is None:
        config = get_config()

    print(f"\n{'='*60}")
    print(f"Evaluating PatchCore on: {category}")
    print(f"{'='*60}")

    # Create output directory
    output_path = Path(config.eval.results_dir) / category
    output_path.mkdir(parents=True, exist_ok=True)

    if config.eval.save_anomaly_maps:
        anomaly_maps_dir = output_path / 'anomaly_maps'
        anomaly_maps_dir.mkdir(exist_ok=True)

    # Load memory bank
    memory_bank_path = Path(config.train.checkpoint_dir) / f"{category}_memory_bank.npy"
    if not memory_bank_path.exists():
        raise FileNotFoundError(f"Memory bank not found: {memory_bank_path}")

    print(f"\nLoading memory bank from {memory_bank_path}")

    model = PatchCore(
        backbone=config.model.backbone,
        device=config.eval.device,
        config=config.model
    )
    model.load(str(memory_bank_path))

    # Create test dataloader
    print(f"\nLoading test data...")
    test_loader = get_dataloader(
        root=config.data.root_path,
        category=category,
        split='test',
        batch_size=config.eval.batch_size,
        num_workers=config.data.num_workers,
        shuffle=False
    )
    print(f"  Test samples: {len(test_loader.dataset)}")

    # Collect predictions and ground truth
    all_scores = []
    all_labels = []
    all_anomaly_maps = []
    all_masks = []
    all_paths = []

    print(f"\nRunning inference...")
    start_time = time.time()

    idx = 0
    for images, labels, masks in tqdm(test_loader, desc="  Inference"):
        # Get batch size
        B = images.shape[0]

        # Predict
        scores, anomaly_maps = model.predict(images)

        # Apply Gaussian smoothing to anomaly maps
        for i in range(B):
            anomaly_maps[i] = gaussian_filter(
                anomaly_maps[i],
                sigma=config.eval.anomaly_map_sigma
            )

        # Collect results
        all_scores.extend(scores.tolist())
        all_labels.extend(labels.tolist())
        all_anomaly_maps.extend(anomaly_maps)
        all_masks.extend(masks.numpy())

        # Get image paths
        for i in range(B):
            path = test_loader.dataset.get_image_path(idx + i)
            all_paths.append(path)

        idx += B

    elapsed = time.time() - start_time
    print(f"  Inference completed in {elapsed:.1f} seconds")
    print(f"  FPS: {len(all_scores) / elapsed:.1f}")

    # Calculate metrics
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    all_anomaly_maps = np.array(all_anomaly_maps)
    all_masks = np.array(all_masks).squeeze(1)  # [N, H, W]

    # Image-level AUROC
    image_auroc = roc_auc_score(all_labels, all_scores)
    print(f"\n  Image-level AUROC: {image_auroc*100:.1f}%")

    # Pixel-level AUROC (only for anomaly samples with masks)
    pixel_auroc = None
    anomaly_indices = np.where(all_labels == 1)[0]

    if len(anomaly_indices) > 0:
        # Flatten masks and anomaly maps for anomaly samples
        flat_masks = all_masks[anomaly_indices].flatten()
        flat_maps = all_anomaly_maps[anomaly_indices].flatten()

        # Only calculate if there are positive pixels
        if flat_masks.sum() > 0:
            pixel_auroc = roc_auc_score(flat_masks, flat_maps)
            print(f"  Pixel-level AUROC: {pixel_auroc*100:.1f}%")
        else:
            print("  Pixel-level AUROC: N/A (no ground truth masks)")

    # Save anomaly maps
    if config.eval.save_anomaly_maps:
        print(f"\n  Saving anomaly maps...")
        for i, (path, amap) in enumerate(zip(all_paths, all_anomaly_maps)):
            # Save raw anomaly map
            filename = Path(path).stem + '_anomaly.npy'
            np.save(str(anomaly_maps_dir / filename), amap)

    # Save per-image scores
    scores_data = {
        'paths': all_paths,
        'scores': all_scores.tolist(),
        'labels': all_labels.tolist()
    }
    with open(output_path / 'scores.json', 'w') as f:
        json.dump(scores_data, f, indent=2)

    # Compile results
    results = {
        'category': category,
        'image_auroc': float(image_auroc),
        'pixel_auroc': float(pixel_auroc) if pixel_auroc is not None else None,
        'num_test_samples': len(all_labels),
        'num_normal': int((all_labels == 0).sum()),
        'num_anomaly': int((all_labels == 1).sum()),
        'inference_time': elapsed,
        'fps': len(all_scores) / elapsed
    }

    # Save metrics
    with open(output_path / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def evaluate_all_categories(config: Config = None) -> dict:
    """
    Evaluate PatchCore on all MVTec AD categories.

    Args:
        config: Config instance (uses default if None)

    Returns:
        Dictionary with all results
    """
    if config is None:
        config = get_config()

    categories = config.data.categories

    print(f"\n{'#'*60}")
    print(f"Evaluating PatchCore on ALL {len(categories)} categories")
    print(f"{'#'*60}")

    all_results = {}
    total_start = time.time()

    for i, category in enumerate(categories, 1):
        print(f"\n[{i}/{len(categories)}]", end="")

        try:
            results = evaluate_category(category, config)
            all_results[category] = results
        except Exception as e:
            print(f"  Error evaluating {category}: {e}")
            all_results[category] = {'error': str(e)}

    total_elapsed = time.time() - total_start

    # Calculate averages
    image_aurocs = [r['image_auroc'] for r in all_results.values() if 'image_auroc' in r]
    pixel_aurocs = [r['pixel_auroc'] for r in all_results.values() if r.get('pixel_auroc') is not None]

    avg_image_auroc = np.mean(image_aurocs) if image_aurocs else 0
    avg_pixel_auroc = np.mean(pixel_aurocs) if pixel_aurocs else 0

    # Print summary
    print(f"\n{'#'*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'#'*60}")
    print(f"\n{'Category':<15} {'Image AUROC':<15} {'Pixel AUROC':<15}")
    print(f"{'-'*45}")

    for category in categories:
        if category in all_results and 'image_auroc' in all_results[category]:
            img_auc = all_results[category]['image_auroc'] * 100
            pix_auc = all_results[category].get('pixel_auroc')
            pix_str = f"{pix_auc*100:.1f}%" if pix_auc else "N/A"
            print(f"{category:<15} {img_auc:<15.1f} {pix_str:<15}")
        else:
            print(f"{category:<15} {'ERROR':<15} {'ERROR':<15}")

    print(f"{'-'*45}")
    print(f"{'AVERAGE':<15} {avg_image_auroc*100:<15.1f} {avg_pixel_auroc*100:<15.1f}")
    print(f"\nTotal evaluation time: {total_elapsed/60:.1f} minutes")

    # Save summary
    summary = {
        'categories': all_results,
        'average_image_auroc': float(avg_image_auroc),
        'average_pixel_auroc': float(avg_pixel_auroc),
        'total_time': total_elapsed
    }

    with open(Path(config.eval.results_dir) / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {config.eval.results_dir}")
    print(f"{'#'*60}")

    return summary


def main():
    # Get default config
    cfg = get_config()

    parser = argparse.ArgumentParser(
        description='Evaluate PatchCore on MVTec AD dataset'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default=cfg.data.root_path,
        help='Path to MVTec AD dataset'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default=cfg.train.checkpoint_dir,
        help='Directory with saved memory banks'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=cfg.eval.results_dir,
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--category',
        type=str,
        default='all',
        choices=['all'] + MVTEC_CATEGORIES,
        help='Category to evaluate (default: all)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=cfg.eval.batch_size,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=cfg.eval.device,
        choices=['cuda', 'cpu'],
        help='Device to use'
    )
    parser.add_argument(
        '--no_save_maps',
        action='store_true',
        help='Do not save anomaly maps'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=cfg.data.num_workers,
        help='Number of data loading workers'
    )

    args = parser.parse_args()

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    if args.device == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Update config with CLI arguments
    cfg.data.root_path = args.data_path
    cfg.train.checkpoint_dir = args.checkpoint_dir
    cfg.eval.results_dir = args.output_dir
    cfg.eval.batch_size = args.batch_size
    cfg.eval.device = args.device
    cfg.eval.save_anomaly_maps = not args.no_save_maps
    cfg.data.num_workers = args.num_workers

    # Evaluate
    if args.category == 'all':
        evaluate_all_categories(cfg)
    else:
        evaluate_category(args.category, cfg)


if __name__ == '__main__':
    main()
