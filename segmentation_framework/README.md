# Segmentation Framework for Industrial Inspection

A professional-grade segmentation framework designed for SWIR Vitrox board inspection, supporting multiple architectures (YOLOv8-Seg, SegFormer, SAM) with curriculum learning and comprehensive augmentation strategies.

## Features

- **Multiple Architectures**: Support for YOLOv8-Seg, SegFormer, and SAM
- **Curriculum Learning**: Dynamic augmentations that evolve during training
- **Industrial-Grade Augmentations**: 
  - Static: Rotation (0°, 90°, 180°, 270°), Flip (X, Y, XY), Random crop
  - Dynamic: Small shifts, rotations, blur, sharpening, intensity/contrast adjustments
  - Advanced: Elastic transforms, grid distortion, noise, motion blur, cutout
- **Mirror Border Mode**: Prevents edge artifacts in geometric transformations
- **Configurable Everything**: Learning rate, input size, augmentation parameters
- **CPU/GPU Support**: Optimized for both CPU and GPU training
- **Complete Pipeline**: Data loading, training, evaluation, visualization, inference

## Installation

```bash
cd segmentation_framework
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

For SAM support:
```bash
pip install -e ".[sam]"
```

## Quick Start

### 1. Prepare Your Data

Organize your data in the `data/` directory:
```
data/
├── raw/          # Original images from CVAT
├── processed/    # Preprocessed data
└── splits/       # Train/val/test splits
```

### 2. Configure Training

Edit `configs/yolo.yaml` or `configs/segformer.yaml`:

```yaml
model:
  name: "yolov8n-seg"  # or "segformer-b0"
  
training:
  epochs: 100
  batch_size: 8
  learning_rate: 0.01
  input_size: 640
  
augmentation:
  curriculum_epoch: 2  # Apply dynamic augmentations every N epochs
  static:
    rotations: [0, 90, 180, 270]
    flips: ["x", "y", "xy"]
    random_crop: true
  dynamic:
    shift_range: 0.1
    rotation_range: 15
    blur_limit: 3
    sharpen_alpha: 0.2
    intensity_offset: 0.1
    contrast_scale: 0.1
```

### 3. Train a Model

```bash
# Using CLI
segtrain train --config configs/yolo.yaml

# Or using Python
python -m segmentation.cli.main train --config configs/yolo.yaml
```

### 4. Evaluate

```bash
segtrain evaluate --config configs/yolo.yaml --checkpoint outputs/checkpoints/best.pt
```

### 5. Inference

```python
from segmentation.inference.predictor import Predictor

predictor = Predictor("outputs/checkpoints/best.pt")
prediction = predictor.predict("path/to/image.jpg")
```

## Directory Structure

```
segmentation_framework/
├── configs/           # Configuration files
├── data/             # Data directories
├── outputs/          # Training outputs
├── src/segmentation/ # Source code
│   ├── config/       # Configuration management
│   ├── datasets/     # Dataset loaders
│   ├── augmentation/ # Augmentation pipelines
│   ├── models/       # Model implementations
│   ├── training/     # Training logic
│   ├── evaluation/   # Metrics & evaluation
│   ├── inference/    # Prediction utilities
│   └── utils/        # Helper functions
└── tests/            # Unit tests
```

## Classes

Default classes for Vitrox board inspection:
- `background` (0)
- `die` (1)
- `scratches` (2)
- `contamination` (3)
- `other_defects` (4)

## Tutorials

See detailed tutorials in the `tutorials/` directory:
- `yolo_tutorial.md` - Complete YOLOv8-Seg guide
- `segformer_tutorial.md` - Complete SegFormer guide
- `QUICK_START.md` - Quick reference

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.
