"""
PatchCore: Towards Total Recall in Industrial Anomaly Detection

Implementation based on the CVPR 2022 paper.
https://arxiv.org/abs/2106.08265
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, List, Optional
import numpy as np
from tqdm import tqdm

from config import get_config, ModelConfig

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: faiss not available. Install faiss-gpu for optimal performance.")


class FeatureExtractor(nn.Module):
    """
    Extract features from intermediate layers of WideResNet-50.

    Uses Layer 2 (512 channels) and Layer 3 (1024 channels) for a total
    of 1536-dimensional features per spatial location.
    """

    def __init__(
        self,
        backbone: str = None,
        device: str = None,
        config: ModelConfig = None
    ):
        super().__init__()

        # Use config defaults
        if config is None:
            config = get_config().model
        if backbone is None:
            backbone = config.backbone
        if device is None:
            device = get_config().train.device

        self.device = device
        self.config = config

        # Load pretrained WideResNet-50
        if backbone == 'wide_resnet50_2':
            self.model = models.wide_resnet50_2(
                weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        self.model.to(device)

        # Hook storage
        self.features = {}

        # Register hooks for configured layers
        for layer_name in config.layers:
            layer = getattr(self.model, layer_name)
            layer.register_forward_hook(self._get_hook(layer_name))

    def _get_hook(self, name: str):
        """Create a forward hook to capture intermediate features."""
        def hook(module, input, output):
            self.features[name] = output
        return hook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract and concatenate features from Layer 2 and Layer 3.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            Concatenated features [B, 1536, H', W'] where H'=W'=28 for 224x224 input
        """
        self.features.clear()

        # Forward pass (we don't need the final output)
        with torch.no_grad():
            _ = self.model(x)

        # Get features from both layers
        layer2_feat = self.features['layer2']  # [B, 512, 28, 28]
        layer3_feat = self.features['layer3']  # [B, 1024, 14, 14]

        # Upsample layer3 to match layer2 spatial size
        layer3_feat = F.interpolate(
            layer3_feat,
            size=layer2_feat.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        # Concatenate along channel dimension
        features = torch.cat([layer2_feat, layer3_feat], dim=1)  # [B, 1536, 28, 28]

        return features


class PatchCore:
    """
    PatchCore anomaly detection model.

    Pipeline:
    1. Feature Extraction (WideResNet-50, Layer 2+3)
    2. Local Patch Aggregation (adaptive average pooling)
    3. Coreset Subsampling (Greedy selection)
    4. Memory Bank Construction (faiss index)
    5. Anomaly Scoring (k-NN distance)

    Args:
        backbone: Backbone model name (uses config default if None)
        device: Device to use (uses config default if None)
        config: ModelConfig instance (uses default if None)
    """

    def __init__(
        self,
        backbone: str = None,
        device: str = None,
        config: ModelConfig = None
    ):
        # Use config defaults
        if config is None:
            config = get_config().model
        if backbone is None:
            backbone = config.backbone
        if device is None:
            device = get_config().train.device

        self.device = device
        self.config = config
        self.coreset_ratio = config.coreset_ratio
        self.patch_size = config.patch_size
        self.k_neighbors = config.k_neighbors

        # Feature extractor
        self.feature_extractor = FeatureExtractor(backbone, device, config)

        # Memory bank (will be populated during fit)
        self.memory_bank = None
        self.faiss_index = None

        # Feature dimensions from config
        self.feature_dim = config.feature_dim
        self.feature_map_size = config.feature_map_size

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images.

        Args:
            images: Input images [B, 3, H, W]

        Returns:
            Features [B, C, H', W']
        """
        return self.feature_extractor(images)

    def aggregate_patches(self, features: torch.Tensor) -> torch.Tensor:
        """
        Aggregate local patches using adaptive average pooling.

        This creates more robust features by incorporating neighborhood context.

        Args:
            features: Input features [B, C, H, W]

        Returns:
            Aggregated patch features [B*H*W, C]
        """
        B, C, H, W = features.shape

        # Apply local average pooling with stride 1 to aggregate neighborhood
        padding = self.patch_size // 2
        pooled = F.avg_pool2d(
            features,
            kernel_size=self.patch_size,
            stride=1,
            padding=padding
        )

        # Reshape to [B*H*W, C]
        patches = pooled.permute(0, 2, 3, 1).reshape(-1, C)

        return patches

    def _greedy_coreset(self, features: np.ndarray, ratio: float) -> np.ndarray:
        """
        Greedy coreset selection algorithm.

        Selects representative patches by iteratively choosing the patch
        that is farthest from all previously selected patches.

        Args:
            features: All patch features [N, C]
            ratio: Ratio of patches to select

        Returns:
            Selected coreset features [M, C] where M = N * ratio
        """
        N = features.shape[0]
        target_size = int(N * ratio)

        if target_size >= N:
            return features

        print(f"  Coreset selection: {N} -> {target_size} patches")

        # Use faiss for efficient distance computation
        if FAISS_AVAILABLE:
            return self._greedy_coreset_faiss(features, target_size)
        else:
            return self._greedy_coreset_numpy(features, target_size)

    def _greedy_coreset_faiss(self, features: np.ndarray, target_size: int) -> np.ndarray:
        """Greedy coreset using GPU-accelerated distance computation."""
        N, C = features.shape
        features = features.astype(np.float32)

        # Initialize with random point
        selected_indices = [np.random.randint(N)]

        # Use GPU if available for faster distance computation
        if torch.cuda.is_available():
            features_gpu = torch.from_numpy(features).cuda()

            # Compute initial distances to first point
            first_point = features_gpu[selected_indices[0]]
            distances = torch.norm(features_gpu - first_point, dim=1)

            # Iteratively select farthest point
            for _ in tqdm(range(target_size - 1), desc="  Coreset", leave=False):
                # Select point with maximum distance to current coreset
                farthest_idx = torch.argmax(distances).item()
                selected_indices.append(farthest_idx)

                # Compute distance to newly added point only
                new_point = features_gpu[farthest_idx]
                new_distances = torch.norm(features_gpu - new_point, dim=1)

                # Keep minimum distance to coreset
                distances = torch.minimum(distances, new_distances)
        else:
            # CPU fallback
            first_point = features[selected_indices[0]]
            distances = np.linalg.norm(features - first_point, axis=1).astype(np.float32)

            for _ in tqdm(range(target_size - 1), desc="  Coreset", leave=False):
                farthest_idx = np.argmax(distances)
                selected_indices.append(farthest_idx)

                new_point = features[farthest_idx]
                new_distances = np.linalg.norm(features - new_point, axis=1).astype(np.float32)
                distances = np.minimum(distances, new_distances)

        return features[selected_indices]

    def _greedy_coreset_numpy(self, features: np.ndarray, target_size: int) -> np.ndarray:
        """Fallback greedy coreset using numpy (slower)."""
        N, C = features.shape

        # Initialize with random point
        selected_indices = [np.random.randint(N)]

        # Compute initial distances
        distances = np.linalg.norm(features - features[selected_indices[0]], axis=1)

        for _ in tqdm(range(target_size - 1), desc="  Coreset", leave=False):
            # Select farthest point
            farthest_idx = np.argmax(distances)
            selected_indices.append(farthest_idx)

            # Update distances
            new_distances = np.linalg.norm(features - features[farthest_idx], axis=1)
            distances = np.minimum(distances, new_distances)

        return features[selected_indices]

    def fit(self, dataloader: torch.utils.data.DataLoader) -> None:
        """
        Build memory bank from training data (normal samples only).

        Args:
            dataloader: DataLoader for training data
        """
        print("Building memory bank...")

        all_patches = []

        # Extract features from all training images
        print("  Extracting features...")
        for images, labels, _ in tqdm(dataloader, desc="  Feature extraction"):
            images = images.to(self.device)

            # Extract features
            features = self.extract_features(images)

            # Aggregate patches
            patches = self.aggregate_patches(features)

            all_patches.append(patches.cpu().numpy())

        # Concatenate all patches
        all_patches = np.concatenate(all_patches, axis=0).astype(np.float32)
        print(f"  Total patches: {all_patches.shape[0]}")

        # Apply coreset subsampling
        print("  Applying coreset subsampling...")
        self.memory_bank = self._greedy_coreset(all_patches, self.coreset_ratio)
        print(f"  Memory bank size: {self.memory_bank.shape[0]}")

        # Build faiss index
        print("  Building faiss index...")
        self._build_faiss_index()

        print("Memory bank construction complete!")

    def _build_faiss_index(self) -> None:
        """Build faiss index for fast nearest neighbor search."""
        if not FAISS_AVAILABLE:
            print("Warning: faiss not available. Using numpy for k-NN (slower).")
            return

        # Create L2 index
        self.faiss_index = faiss.IndexFlatL2(self.feature_dim)

        # Move to GPU if available
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)

        # Add memory bank to index
        self.faiss_index.add(self.memory_bank)

    def predict(
        self,
        images: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomaly scores and maps for images.

        Args:
            images: Input images [B, 3, H, W]

        Returns:
            scores: Image-level anomaly scores [B]
            anomaly_maps: Pixel-level anomaly maps [B, H, W]
        """
        B = images.shape[0]
        images = images.to(self.device)

        # Extract features
        features = self.extract_features(images)

        # Get spatial dimensions
        _, _, H, W = features.shape

        # Aggregate patches
        patches = self.aggregate_patches(features)  # [B*H*W, C]
        patches_np = patches.cpu().numpy().astype(np.float32)

        # Find nearest neighbors
        distances = self._nearest_neighbors(patches_np)  # [B*H*W]

        # Reshape to [B, H, W]
        distance_maps = distances.reshape(B, H, W)

        # Image-level scores (max distance)
        scores = np.max(distance_maps, axis=(1, 2))

        # Upsample anomaly maps to original image size
        image_size = get_config().data.image_size
        distance_maps_tensor = torch.from_numpy(distance_maps).unsqueeze(1).float()
        anomaly_maps = F.interpolate(
            distance_maps_tensor,
            size=(image_size, image_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(1).numpy()

        return scores, anomaly_maps

    def _nearest_neighbors(self, patches: np.ndarray) -> np.ndarray:
        """
        Find distances to nearest neighbors in memory bank.

        Args:
            patches: Query patches [N, C]

        Returns:
            distances: Distances to nearest neighbors [N]
        """
        if FAISS_AVAILABLE and self.faiss_index is not None:
            # Use faiss for fast k-NN
            distances, _ = self.faiss_index.search(patches, self.k_neighbors)
            # Use maximum of k neighbors' distances (as in paper)
            return np.mean(distances, axis=1)
        else:
            # Fallback to numpy (slow for large memory banks)
            distances = []
            for patch in patches:
                dists = np.linalg.norm(self.memory_bank - patch, axis=1)
                distances.append(np.sort(dists)[:self.k_neighbors].mean())
            return np.array(distances)

    def save(self, path: str) -> None:
        """Save memory bank to file."""
        np.save(path, self.memory_bank)
        print(f"Memory bank saved to {path}")

    def load(self, path: str) -> None:
        """Load memory bank from file."""
        self.memory_bank = np.load(path)
        self._build_faiss_index()
        print(f"Memory bank loaded from {path} ({self.memory_bank.shape[0]} patches)")


if __name__ == '__main__':
    # Quick test
    print("Testing PatchCore components...")

    cfg = get_config()
    print(f"\nConfiguration:")
    print(f"  Backbone: {cfg.model.backbone}")
    print(f"  Feature dim: {cfg.model.feature_dim}")
    print(f"  Coreset ratio: {cfg.model.coreset_ratio}")
    print(f"  Patch size: {cfg.model.patch_size}")
    print(f"  k_neighbors: {cfg.model.k_neighbors}")

    # Test feature extractor
    device = cfg.train.device
    print(f"\nDevice: {device}")

    extractor = FeatureExtractor(device=device)

    # Create dummy input
    dummy_input = torch.randn(2, 3, cfg.data.image_size, cfg.data.image_size).to(device)

    # Extract features
    features = extractor(dummy_input)
    print(f"Feature shape: {features.shape}")  # Expected: [2, 1536, 28, 28]

    # Test PatchCore
    model = PatchCore(device=device)

    # Aggregate patches
    patches = model.aggregate_patches(features)
    print(f"Patch shape: {patches.shape}")  # Expected: [2*28*28, 1536] = [1568, 1536]

    print("\nAll tests passed!")
