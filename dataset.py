"""
MVTec AD Dataset Loader for PatchCore

Supports all 15 categories with train/test splits and ground truth masks.
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional, Union

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

from config import get_config, DataConfig


# Get default config
_cfg = get_config()

# MVTec AD Categories (from config)
MVTEC_CATEGORIES = _cfg.data.categories


class MVTecDataset(Dataset):
    """
    MVTec Anomaly Detection Dataset

    Directory structure:
    root/
    ├── bottle/
    │   ├── train/
    │   │   └── good/
    │   ├── test/
    │   │   ├── good/
    │   │   ├── broken_large/
    │   │   └── ...
    │   └── ground_truth/
    │       ├── broken_large/
    │       └── ...
    └── ...

    Args:
        root: Path to MVTec AD dataset root
        category: Category name (e.g., 'bottle', 'cable')
        split: 'train' or 'test'
        config: DataConfig instance (uses default if None)
        transform: Optional custom transform
    """

    def __init__(
        self,
        root: str = None,
        category: str = 'bottle',
        split: str = 'train',
        config: DataConfig = None,
        transform: Optional[transforms.Compose] = None
    ):
        # Use config defaults if not specified
        if config is None:
            config = get_config().data

        if root is None:
            root = config.root_path

        assert category in MVTEC_CATEGORIES, f"Unknown category: {category}"
        assert split in ['train', 'test'], f"Split must be 'train' or 'test'"

        self.root = Path(root)
        self.category = category
        self.split = split
        self.image_size = config.image_size
        self.config = config

        # Default transform for PatchCore
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=config.mean,
                    std=config.std
                )
            ])
        else:
            self.transform = transform

        # Mask transform (no normalization)
        self.mask_transform = transforms.Compose([
            transforms.Resize(
                (self.image_size, self.image_size),
                interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.ToTensor()
        ])

        # Load samples
        self.samples = self._load_samples()

    def _load_samples(self) -> List[dict]:
        """Load all samples for the category and split."""
        samples = []
        category_path = self.root / self.category / self.split

        if not category_path.exists():
            raise FileNotFoundError(f"Category path not found: {category_path}")

        # Iterate through defect types (or 'good' for normal)
        for defect_type in sorted(category_path.iterdir()):
            if not defect_type.is_dir():
                continue

            defect_name = defect_type.name
            is_normal = (defect_name == 'good')

            # Get corresponding mask directory (only for test split anomalies)
            mask_dir = None
            if self.split == 'test' and not is_normal:
                mask_dir = self.root / self.category / 'ground_truth' / defect_name

            # Load images
            for img_path in sorted(defect_type.glob('*.png')):
                sample = {
                    'image_path': str(img_path),
                    'label': 0 if is_normal else 1,  # 0: normal, 1: anomaly
                    'defect_type': defect_name,
                    'mask_path': None
                }

                # Find corresponding mask
                if mask_dir is not None and mask_dir.exists():
                    mask_name = img_path.stem + '_mask.png'
                    mask_path = mask_dir / mask_name
                    if mask_path.exists():
                        sample['mask_path'] = str(mask_path)

                samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Returns:
            image: Transformed image tensor [C, H, W]
            label: 0 for normal, 1 for anomaly
            mask: Ground truth mask [1, H, W] (zeros if no mask)
        """
        sample = self.samples[idx]

        # Load and transform image
        image = Image.open(sample['image_path']).convert('RGB')
        image = self.transform(image)

        # Load mask if available
        if sample['mask_path'] is not None:
            mask = Image.open(sample['mask_path']).convert('L')
            mask = self.mask_transform(mask)
            mask = (mask > 0.5).float()  # Binarize
        else:
            # No mask: zeros for normal, or missing mask
            mask = torch.zeros(1, self.image_size, self.image_size)

        return image, sample['label'], mask

    def get_image_path(self, idx: int) -> str:
        """Get the file path of an image by index."""
        return self.samples[idx]['image_path']

    def get_defect_type(self, idx: int) -> str:
        """Get the defect type of a sample by index."""
        return self.samples[idx]['defect_type']


def get_dataloader(
    root: str = None,
    category: str = 'bottle',
    split: str = 'train',
    batch_size: int = None,
    num_workers: int = None,
    shuffle: bool = None,
    config: DataConfig = None
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for MVTec AD dataset.

    Args:
        root: Path to MVTec AD dataset (uses config default if None)
        category: Category name
        split: 'train' or 'test'
        batch_size: Batch size (uses config default if None)
        num_workers: Number of data loading workers (uses config default if None)
        shuffle: Whether to shuffle (default: True for train, False for test)
        config: DataConfig instance (uses default if None)

    Returns:
        DataLoader instance
    """
    # Use config defaults
    if config is None:
        config = get_config().data

    if root is None:
        root = config.root_path
    if batch_size is None:
        batch_size = get_config().train.batch_size
    if num_workers is None:
        num_workers = config.num_workers

    dataset = MVTecDataset(
        root=root,
        category=category,
        split=split,
        config=config
    )

    if shuffle is None:
        shuffle = (split == 'train')

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )


if __name__ == '__main__':
    # Quick test
    import argparse

    cfg = get_config()

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=cfg.data.root_path)
    parser.add_argument('--category', type=str, default='bottle')
    args = parser.parse_args()

    print(f"Testing MVTec dataset loader...")
    print(f"Data path: {args.data_path}")
    print(f"Category: {args.category}")
    print(f"Image size: {cfg.data.image_size}")
    print(f"Available categories: {MVTEC_CATEGORIES}")

    # Test train split
    train_dataset = MVTecDataset(args.data_path, args.category, 'train')
    print(f"\nTrain samples: {len(train_dataset)}")

    # Test test split
    test_dataset = MVTecDataset(args.data_path, args.category, 'test')
    print(f"Test samples: {len(test_dataset)}")

    # Count labels
    labels = [s['label'] for s in test_dataset.samples]
    print(f"  - Normal: {labels.count(0)}")
    print(f"  - Anomaly: {labels.count(1)}")

    # Test loading
    image, label, mask = train_dataset[0]
    print(f"\nSample shapes:")
    print(f"  - Image: {image.shape}")
    print(f"  - Label: {label}")
    print(f"  - Mask: {mask.shape}")
