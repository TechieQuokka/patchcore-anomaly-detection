"""
PatchCore Configuration

All hyperparameters and settings in one place.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class DataConfig:
    """Dataset configuration"""
    # Path to MVTec AD dataset
    root_path: str = '/mnt/e/Big Data/MVTec AD'

    # Input image size
    image_size: int = 224

    # Number of data loading workers
    num_workers: int = 4

    # All MVTec AD categories
    categories: List[str] = field(default_factory=lambda: [
        # Objects (10)
        'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
        'pill', 'screw', 'toothbrush', 'transistor', 'zipper',
        # Textures (5)
        'carpet', 'grid', 'leather', 'tile', 'wood'
    ])

    # ImageNet normalization
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Backbone network
    backbone: str = 'wide_resnet50_2'

    # Feature extraction layers
    layers: List[str] = field(default_factory=lambda: ['layer2', 'layer3'])

    # Feature dimensions
    # layer2: 512, layer3: 1024 -> total: 1536
    layer2_dim: int = 512
    layer3_dim: int = 1024
    feature_dim: int = 1536  # layer2_dim + layer3_dim

    # Feature map size for 224x224 input
    feature_map_size: int = 28

    # Local patch aggregation kernel size
    patch_size: int = 3

    # Number of neighbors for k-NN scoring
    k_neighbors: int = 9

    # Coreset subsampling ratio (25% as in paper)
    coreset_ratio: float = 0.25


@dataclass
class TrainConfig:
    """Training configuration"""
    # Batch size for feature extraction
    batch_size: int = 128

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Output directory for memory banks
    checkpoint_dir: str = './checkpoints'


@dataclass
class EvalConfig:
    """Evaluation configuration"""
    # Batch size for inference
    batch_size: int = 128

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Output directory for results
    results_dir: str = './results'

    # Whether to save anomaly maps
    save_anomaly_maps: bool = True

    # Gaussian smoothing sigma for anomaly maps
    anomaly_map_sigma: float = 4.0


@dataclass
class Config:
    """Main configuration combining all sub-configs"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


# Default configuration instance
def get_config() -> Config:
    """Get default configuration"""
    return Config()


def get_config_dict() -> dict:
    """Get configuration as dictionary"""
    cfg = get_config()
    return {
        'data': {
            'root_path': cfg.data.root_path,
            'image_size': cfg.data.image_size,
            'num_workers': cfg.data.num_workers,
            'categories': cfg.data.categories,
            'mean': cfg.data.mean,
            'std': cfg.data.std,
        },
        'model': {
            'backbone': cfg.model.backbone,
            'layers': cfg.model.layers,
            'feature_dim': cfg.model.feature_dim,
            'feature_map_size': cfg.model.feature_map_size,
            'patch_size': cfg.model.patch_size,
            'k_neighbors': cfg.model.k_neighbors,
            'coreset_ratio': cfg.model.coreset_ratio,
        },
        'train': {
            'batch_size': cfg.train.batch_size,
            'device': cfg.train.device,
            'checkpoint_dir': cfg.train.checkpoint_dir,
        },
        'eval': {
            'batch_size': cfg.eval.batch_size,
            'device': cfg.eval.device,
            'results_dir': cfg.eval.results_dir,
            'save_anomaly_maps': cfg.eval.save_anomaly_maps,
            'anomaly_map_sigma': cfg.eval.anomaly_map_sigma,
        }
    }


if __name__ == '__main__':
    # Print configuration
    cfg = get_config()

    print("="*60)
    print("PatchCore Configuration")
    print("="*60)

    print("\n[Data Config]")
    print(f"  root_path: {cfg.data.root_path}")
    print(f"  image_size: {cfg.data.image_size}")
    print(f"  num_workers: {cfg.data.num_workers}")
    print(f"  categories: {len(cfg.data.categories)} categories")

    print("\n[Model Config]")
    print(f"  backbone: {cfg.model.backbone}")
    print(f"  layers: {cfg.model.layers}")
    print(f"  feature_dim: {cfg.model.feature_dim}")
    print(f"  patch_size: {cfg.model.patch_size}")
    print(f"  k_neighbors: {cfg.model.k_neighbors}")
    print(f"  coreset_ratio: {cfg.model.coreset_ratio}")

    print("\n[Train Config]")
    print(f"  batch_size: {cfg.train.batch_size}")
    print(f"  device: {cfg.train.device}")
    print(f"  checkpoint_dir: {cfg.train.checkpoint_dir}")

    print("\n[Eval Config]")
    print(f"  batch_size: {cfg.eval.batch_size}")
    print(f"  device: {cfg.eval.device}")
    print(f"  results_dir: {cfg.eval.results_dir}")
    print(f"  save_anomaly_maps: {cfg.eval.save_anomaly_maps}")
    print(f"  anomaly_map_sigma: {cfg.eval.anomaly_map_sigma}")
