# PatchCore Architecture

> **Towards Total Recall in Industrial Anomaly Detection** (CVPR 2022)

---

## Overview

PatchCore는 사전 학습된 CNN의 로컬 패치 특징을 메모리 뱅크에 저장하고, 테스트 시 가장 가까운 정상 패치와의 거리로 이상을 탐지하는 방식입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                         PatchCore                               │
│                                                                 │
│   "정상 이미지의 로컬 패치 특징을 저장해두고,                   │
│    테스트 시 가장 가까운 정상 패치와의 거리로 이상 탐지"        │
│                                                                 │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐      │
│   │   Feature   │     │   Memory    │     │   Anomaly   │      │
│   │  Extractor  │ ──▶ │    Bank     │ ──▶ │   Scoring   │      │
│   │ (WideRes50) │     │  (Coreset)  │     │   (k-NN)    │      │
│   └─────────────┘     └─────────────┘     └─────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
patch-core/2026-02-04/
├── docs/
│   └── architecture.md     # 아키텍처 문서 (현재 파일)
├── patchcore.py            # PatchCore 클래스 (핵심 로직)
├── dataset.py              # MVTec AD 데이터 로더
├── train.py                # 학습 스크립트 (feature 추출 + memory bank)
├── evaluate.py             # 평가 스크립트 (AUROC 계산)
├── visualization.ipynb     # 시각화 노트북
└── requirements.txt        # 의존성
```

---

## Configuration

| 항목 | 값 | 비고 |
|------|-----|------|
| **Backbone** | WideResNet-50 | ImageNet pretrained |
| **Feature Layers** | Layer 2 + Layer 3 | 중간 레이어 조합 |
| **Input Size** | 224 × 224 | Feature map: 28 × 28 |
| **Batch Size** | 32 | RTX 3060 12GB 기준 |
| **Coreset Ratio** | 25% | Greedy Selection |
| **k-NN Library** | faiss-gpu | GPU 가속 |

### Hardware Requirements

```
┌─────────────────────────────────────────────────────────────┐
│  Target Hardware: NVIDIA GeForce RTX 3060 12GB              │
├─────────────────────────────────────────────────────────────┤
│  Component              │  Memory Usage   │  Status         │
├─────────────────────────────────────────────────────────────┤
│  WideResNet-50 Model    │  ~200MB         │  ✅ OK          │
│  Forward Pass (B=32)    │  ~4GB           │  ✅ OK          │
│  Memory Bank (25%)      │  ~16MB/category │  ✅ OK          │
│  faiss-gpu Index        │  ~500MB         │  ✅ OK          │
├─────────────────────────────────────────────────────────────┤
│  Total Peak Usage       │  ~6GB           │  ✅ Headroom    │
└─────────────────────────────────────────────────────────────┘
```

---

## Pipeline Architecture

### Training Phase

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING PHASE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. FEATURE EXTRACTION                                         │
│   ─────────────────────                                         │
│   Normal Images         WideResNet-50              Features     │
│   ┌─────────┐          ┌─────────────┐          ┌─────────┐    │
│   │  224×   │          │             │          │ 512×28× │    │
│   │  224×3  │────────▶ │  Layer 2 ───────────▶  │   28    │    │
│   │         │          │             │          └────┬────┘    │
│   └─────────┘          │  Layer 3 ───────────▶  ┌────┴────┐    │
│                        │             │          │1024×28× │    │
│                        └─────────────┘          │   28    │    │
│                                                 └────┬────┘    │
│                                                      │         │
│   2. FEATURE CONCATENATION                           ▼         │
│   ────────────────────────                    ┌───────────┐    │
│   Layer 2 (512) + Layer 3 (1024)              │ Concat    │    │
│   = 1536 dimensions                           │ 1536×28×  │    │
│                                               │   28      │    │
│                                               └─────┬─────┘    │
│                                                     │          │
│   3. LOCAL PATCH AGGREGATION                        ▼          │
│   ───────────────────────────              ┌─────────────┐     │
│   AdaptiveAvgPool(3×3) per position        │ Aggregated  │     │
│   → 주변 컨텍스트 포함                      │  Patches    │     │
│                                            │ 784 × 1536  │     │
│                                            └──────┬──────┘     │
│                                                   │            │
│   4. CORESET SUBSAMPLING (25%)                    ▼            │
│   ────────────────────────────            ┌─────────────┐      │
│   Greedy Selection Algorithm              │ Memory Bank │      │
│   172K patches → 43K patches              │  (Coreset)  │      │
│                                           │  43K×1536   │      │
│                                           └─────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Test Phase

```
┌─────────────────────────────────────────────────────────────────┐
│                          TEST PHASE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Test Image            Feature Extraction        k-NN Search   │
│   ┌─────────┐          ┌─────────────┐         ┌─────────────┐ │
│   │  224×   │          │  Patches    │         │   faiss     │ │
│   │  224×3  │────────▶ │  784×1536   │────────▶│    GPU      │ │
│   │(defect?)│          │             │         │   Index     │ │
│   └─────────┘          └─────────────┘         └──────┬──────┘ │
│                                                       │        │
│                                                       ▼        │
│   Output                Distance Map            Nearest        │
│   ┌─────────┐          ┌─────────────┐         Distance       │
│   │ Anomaly │          │  28 × 28    │    ┌─────────────┐     │
│   │  Score  │◀─────────│  distances  │◀───│ Memory Bank │     │
│   │ =max(d) │          │             │    └─────────────┘     │
│   └─────────┘          └──────┬──────┘                        │
│                               │                                │
│                               ▼                                │
│                        ┌─────────────┐                         │
│                        │  Upsample   │                         │
│                        │  to 224×224 │                         │
│                        │  (Heatmap)  │                         │
│                        └─────────────┘                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Design

### 1. PatchCore Class (`patchcore.py`)

```
┌─────────────────────────────────────────────────────────────────┐
│  class PatchCore                                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Attributes:                                                    │
│  ───────────                                                    │
│  - backbone: WideResNet50 (frozen)                              │
│  - feature_extractor: Hook-based layer extraction               │
│  - memory_bank: faiss.IndexFlatL2 (GPU)                         │
│  - coreset_ratio: float (0.25)                                  │
│                                                                 │
│  Methods:                                                       │
│  ────────                                                       │
│  + __init__(backbone, device, coreset_ratio)                    │
│  + extract_features(images) → Tensor[B, C, H, W]                │
│  + aggregate_patches(features) → Tensor[B, N, C]                │
│  + fit(dataloader) → None  # Build memory bank                  │
│  + predict(images) → (scores, anomaly_maps)                     │
│  + _greedy_coreset(features, ratio) → Tensor                    │
│  + _nearest_neighbors(patches, k) → distances                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Dataset Class (`dataset.py`)

```
┌─────────────────────────────────────────────────────────────────┐
│  class MVTecDataset(torch.utils.data.Dataset)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Attributes:                                                    │
│  ───────────                                                    │
│  - root_path: str ("/mnt/e/Big Data/MVTec AD")                  │
│  - category: str (e.g., "bottle", "cable", ...)                 │
│  - split: str ("train" | "test")                                │
│  - transform: torchvision.transforms                            │
│                                                                 │
│  Methods:                                                       │
│  ────────                                                       │
│  + __init__(root, category, split, transform)                   │
│  + __len__() → int                                              │
│  + __getitem__(idx) → (image, label, mask)                      │
│                                                                 │
│  Categories (15):                                               │
│  ─────────────────                                              │
│  Objects:  bottle, cable, capsule, hazelnut, metal_nut,         │
│            pill, screw, toothbrush, transistor, zipper          │
│  Textures: carpet, grid, leather, tile, wood                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Training Script (`train.py`)

```
┌─────────────────────────────────────────────────────────────────┐
│  train.py - Memory Bank Construction                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Flow:                                                          │
│  ─────                                                          │
│  1. Load MVTec category (train split, normal only)              │
│  2. Initialize PatchCore model                                  │
│  3. Extract features from all training images                   │
│  4. Apply Greedy Coreset subsampling (25%)                      │
│  5. Build faiss GPU index                                       │
│  6. Save memory bank to disk                                    │
│                                                                 │
│  CLI Interface:                                                 │
│  ──────────────                                                 │
│  python train.py --data_path /mnt/e/Big\ Data/MVTec\ AD \       │
│                  --category bottle \                            │
│                  --coreset_ratio 0.25                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4. Evaluation Script (`evaluate.py`)

```
┌─────────────────────────────────────────────────────────────────┐
│  evaluate.py - Anomaly Detection & Metrics                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Flow:                                                          │
│  ─────                                                          │
│  1. Load trained memory bank                                    │
│  2. Load MVTec category (test split)                            │
│  3. For each test image:                                        │
│     - Extract features                                          │
│     - Find k-NN distances                                       │
│     - Compute anomaly score (max distance)                      │
│     - Generate anomaly map (upsampled distances)                │
│  4. Calculate metrics:                                          │
│     - Image-level AUROC                                         │
│     - Pixel-level AUROC                                         │
│  5. Save results                                                │
│                                                                 │
│  Output:                                                        │
│  ───────                                                        │
│  - scores.json: per-image anomaly scores                        │
│  - metrics.json: AUROC values                                   │
│  - anomaly_maps/: heatmap images                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Algorithm Details

### Greedy Coreset Selection

정상 패치 전체에서 대표 패치를 선택하는 알고리즘:

```
┌─────────────────────────────────────────────────────────────────┐
│  GREEDY CORESET ALGORITHM                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: P = {p_1, ..., p_N}  # All patches (N=172K)             │
│  Output: C ⊂ P               # Coreset (|C| = 0.25N = 43K)      │
│                                                                 │
│  Algorithm:                                                     │
│  ──────────                                                     │
│  1. Initialize C = {random patch from P}                        │
│  2. While |C| < target_size:                                    │
│     a. For each p in P \ C:                                     │
│        - d(p) = min distance to any point in C                  │
│     b. Select p* = argmax d(p)  # Farthest point                │
│     c. Add p* to C                                              │
│  3. Return C                                                    │
│                                                                 │
│  Complexity: O(N × M) where M = target coreset size             │
│                                                                 │
│  Optimization:                                                  │
│  ─────────────                                                  │
│  - Use faiss for batch distance computation                     │
│  - Update distances incrementally                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Anomaly Scoring

```
┌─────────────────────────────────────────────────────────────────┐
│  ANOMALY SCORING                                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  For test image X:                                              │
│                                                                 │
│  1. Extract patches: P_test = {p_1, ..., p_784}                 │
│                                                                 │
│  2. For each patch p_i:                                         │
│     d_i = min ||p_i - m||_2  for m in MemoryBank                │
│                                                                 │
│  3. Image-level score:                                          │
│     S_image = max(d_1, ..., d_784)                              │
│                                                                 │
│  4. Pixel-level anomaly map:                                    │
│     - Reshape distances to 28×28                                │
│     - Upsample to 224×224 (bilinear)                            │
│     - Apply Gaussian smoothing (optional)                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                       DATA FLOW DIAGRAM                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    TRAINING                              │   │
│  │                                                          │   │
│  │  MVTec Train     WideResNet     Patches      Coreset     │   │
│  │  (normal)        (frozen)                                │   │
│  │                                                          │   │
│  │  [224×224×3] ──▶ [1536×28×28] ──▶ [N×1536] ──▶ [M×1536]  │   │
│  │  ~220 images     Layer 2+3       784×220      25% of N   │   │
│  │                                  =172K        =43K       │   │
│  │                                                          │   │
│  │                                         ▼                │   │
│  │                                  ┌─────────────┐         │   │
│  │                                  │ Memory Bank │         │   │
│  │                                  │ (faiss GPU) │         │   │
│  │                                  └──────┬──────┘         │   │
│  └─────────────────────────────────────────│────────────────┘   │
│                                            │ save               │
│                                            ▼                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    INFERENCE                             │   │
│  │                                            │ load            │
│  │                                            ▼                 │
│  │                                  ┌─────────────┐         │   │
│  │                                  │ Memory Bank │         │   │
│  │                                  └──────┬──────┘         │   │
│  │                                         │                │   │
│  │  Test Image      WideResNet     Patches │   k-NN        │   │
│  │  (any)           (frozen)               ▼                │   │
│  │                                                          │   │
│  │  [224×224×3] ──▶ [1536×28×28] ──▶ [784×1536] ──▶ [784]   │   │
│  │   1 image        Layer 2+3       patches      distances  │   │
│  │                                                   │      │   │
│  │                                                   ▼      │   │
│  │                               ┌───────────┬───────────┐  │   │
│  │                               │  max(d)   │  reshape  │  │   │
│  │                               │           │  upsample │  │   │
│  │                               ▼           ▼           │  │   │
│  │                          [Score]    [Anomaly Map]     │  │   │
│  │                           float      224×224          │  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Expected Performance

### Target Metrics (MVTec AD)

| Category | Image AUROC | Pixel AUROC |
|----------|-------------|-------------|
| bottle | 100.0% | 98.6% |
| cable | 99.5% | 98.4% |
| capsule | 98.1% | 98.8% |
| hazelnut | 100.0% | 98.7% |
| metal_nut | 100.0% | 98.4% |
| pill | 96.6% | 97.4% |
| screw | 98.1% | 99.4% |
| toothbrush | 100.0% | 98.7% |
| transistor | 100.0% | 96.3% |
| zipper | 99.4% | 98.8% |
| carpet | 98.7% | 99.0% |
| grid | 98.2% | 98.7% |
| leather | 100.0% | 99.3% |
| tile | 98.7% | 95.6% |
| wood | 99.2% | 95.0% |
| **Average** | **99.1%** | **98.1%** |

---

## Dependencies

```
# requirements.txt
torch>=1.9.0
torchvision>=0.10.0
faiss-gpu>=1.7.0
numpy>=1.19.0
scipy>=1.7.0
scikit-learn>=0.24.0
Pillow>=8.0.0
matplotlib>=3.4.0
tqdm>=4.62.0
```

---

## References

- Paper: [Towards Total Recall in Industrial Anomaly Detection](https://arxiv.org/abs/2106.08265)
- MVTec AD: [MVTec Anomaly Detection Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
