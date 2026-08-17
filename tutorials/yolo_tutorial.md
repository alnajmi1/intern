# YOLOv8-Seg Training Tutorial for SWIR Vitrox Board Inspection

## 🎯 Overview
This tutorial guides you through training a **YOLOv8-Seg** instance segmentation model for inspecting SWIR Vitrox boards. We'll cover data preparation, augmentation strategies with curriculum learning, training, and evaluation.

**Why YOLOv8-Seg?**
- **Fast training** even on CPU (compared to transformers)
- Real-time inference capabilities
- Instance segmentation (detects individual defect instances)
- Excellent balance of speed and accuracy
- Easy deployment and integration
- Built-in augmentation pipeline

---

## 📋 Prerequisites

### System Requirements
- **Hardware**: CPU (as specified), 8GB+ RAM minimum, 16GB recommended
- **Python**: 3.8+
- **Package Manager**: conda

### Installation Steps

```bash
# Create conda environment
conda create -n yolo_inspection python=3.9 -y
conda activate yolo_inspection

# Install Ultralytics YOLOv8
pip install ultralytics

# Install additional dependencies
pip install opencv-python pillow numpy matplotlib tqdm pandas seaborn
pip install scikit-image scipy

# Verify installation
python -c "from ultralytics import YOLO; print('YOLOv8 installed successfully!')"
```

---

## 📁 Data Preparation

### Expected Directory Structure
```
data/
├── raw_images/              # Original images from CVAT
│   ├── img_001.png
│   ├── img_002.png
│   └── ...
├── masks_coco/              # COCO format annotations from CVAT
│   └── annotations.json     # COCO JSON format
├── splits/
│   ├── train.txt            # List of training image filenames
│   ├── val.txt              # List of validation image filenames
│   └── test.txt             # List of test image filenames
└── yolo_dataset/            # Will be created by conversion script
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

### Class Mapping for YOLO
```yaml
# In dataset.yaml file
names:
  0: background  # Note: background is typically not annotated
  1: die
  2: scratches
  3: contamination
  4: other_defects
```

**Important**: In instance segmentation, we only annotate objects (die, scratches, contamination). Background is implicit.

### Exporting from CVAT for YOLO

1. **In CVAT, create your dataset with these labels:**
   - die
   - scratches
   - contamination
   - other_defects (optional)
   
   *Note: Don't annotate "background" - it's implicit*

2. **Export format: COCO 1.0**
   - Go to Menu → Export Dataset
   - Select "COCO 1.0" format
   - Download the zip file

3. **Convert COCO to YOLO format** using the script below

---

## 🔄 Data Conversion Script

Save as `scripts/convert_coco_to_yolo.py`:

```python
#!/usr/bin/env python3
"""
Convert COCO format (from CVAT) to YOLO format for instance segmentation.
"""

import os
import json
import argparse
import shutil
from PIL import Image
import cv2
import numpy as np

def coco_to_yolo(coco_json_path, images_dir, output_dir, classes):
    """
    Convert COCO annotations to YOLO format.
    
    Args:
        coco_json_path: Path to COCO JSON file
        images_dir: Directory containing images
        output_dir: Output directory for YOLO dataset
        classes: List of class names (excluding background)
    """
    
    # Load COCO annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create class name to ID mapping
    class_name_to_id = {name: idx for idx, name in enumerate(classes)}
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)
    
    # Get image info
    images = {img['id']: img for img in coco_data['images']}
    annotations = coco_data['annotations']
    
    # Group annotations by image
    img_annotations = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)
    
    # Process each image
    for img_id, img_info in images.items():
        img_filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Determine if train or val (simple 80/20 split based on ID)
        # You can customize this logic
        is_train = img_id % 5 != 0  # 80% train, 20% val
        
        split = 'train' if is_train else 'val'
        
        # Copy image
        src_img_path = os.path.join(images_dir, img_filename)
        dst_img_path = os.path.join(output_dir, 'images', split, img_filename)
        
        if os.path.exists(src_img_path):
            shutil.copy(src_img_path, dst_img_path)
        else:
            print(f"Warning: Image not found: {src_img_path}")
            continue
        
        # Create YOLO label file
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        label_path = os.path.join(output_dir, 'labels', split, label_filename)
        
        # Process annotations for this image
        yolo_lines = []
        
        if img_id in img_annotations:
            for ann in img_annotations[img_id]:
                category_id = ann['category_id']
                
                # Find class name from COCO categories
                cat_name = None
                for cat in coco_data['categories']:
                    if cat['id'] == category_id:
                        cat_name = cat['name']
                        break
                
                if cat_name not in class_name_to_id:
                    continue  # Skip unknown classes
                
                class_id = class_name_to_id[cat_name]
                
                # Get segmentation mask
                if 'segmentation' in ann and ann['segmentation']:
                    # Handle both RLE and polygon formats
                    seg = ann['segmentation']
                    
                    if isinstance(seg, list) and len(seg) > 0:
                        # Polygon format
                        if isinstance(seg[0], list):
                            # Single polygon
                            polygon = seg[0]
                        else:
                            # Multiple polygons - use first one
                            polygon = seg[0]
                        
                        # Convert to YOLO format (normalized x,y coordinates)
                        if len(polygon) >= 6:  # Minimum 3 points
                            yolo_points = []
                            for i in range(0, len(polygon), 2):
                                x = polygon[i] / img_width
                                y = polygon[i + 1] / img_height
                                yolo_points.extend([x, y])
                            
                            if len(yolo_points) >= 6:
                                line = f"{class_id} " + " ".join([f"{p:.6f}" for p in yolo_points])
                                yolo_lines.append(line)
        
        # Write label file
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
        
        if len(yolo_lines) == 0:
            # Create empty file if no annotations
            pass
    
    print(f"Conversion complete!")
    print(f"Output directory: {output_dir}")


def create_dataset_yaml(output_dir, classes, train_ratio=0.8):
    """Create YOLO dataset YAML configuration."""
    
    yaml_content = f"""# YOLO Dataset Configuration for SWIR Board Inspection

# Dataset paths
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

# Classes
nc: {len(classes)}
names:
"""
    
    for idx, class_name in enumerate(classes):
        yaml_content += f"  {idx}: {class_name}\n"
    
    # Save YAML file
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Dataset YAML created: {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO to YOLO format")
    parser.add_argument("--coco_json", type=str, required=True, 
                       help="Path to COCO JSON file")
    parser.add_argument("--images_dir", type=str, required=True,
                       help="Directory containing images")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for YOLO dataset")
    parser.add_argument("--classes", type=str, nargs='+', 
                       default=['die', 'scratches', 'contamination', 'other_defects'],
                       help="List of class names")
    
    args = parser.parse_args()
    
    coco_to_yolo(args.coco_json, args.images_dir, args.output_dir, args.classes)
    create_dataset_yaml(args.output_dir, args.classes)
```

### Usage:
```bash
python scripts/convert_coco_to_yolo.py \
    --coco_json data/masks_coco/annotations.json \
    --images_dir data/raw_images \
    --output_dir data/yolo_dataset \
    --classes die scratches contamination other_defects
```

---

## 🎨 Augmentation Strategy with Curriculum Learning

### Configuration File

Save as `configs/yolo_config.yaml`:

```yaml
# YOLOv8-Seg Training Configuration for SWIR Board Inspection

# Model Settings
model:
  variant: "yolov8n-seg.pt"  # Options: n, s, m, l, x (nano, small, medium, large, extra-large)
  # For CPU: use yolov8n-seg.pt or yolov8s-seg.pt
  # For better accuracy: yolov8m-seg.pt or larger
  pretrained: true

# Data Settings
data:
  dataset_yaml: "data/yolo_dataset/dataset.yaml"
  input_size: 640  # Default YOLO input size
  batch_size: 4    # Reduce for CPU training
  workers: 2       # Number of data loading workers

# Augmentation Settings
augmentation:
  # Static augmentations (built into YOLO, always applied)
  static:
    hsv_h: 0.015      # Hue augmentation
    hsv_s: 0.7        # Saturation augmentation
    hsv_v: 0.4        # Value augmentation
    degrees: 0.0      # Rotation degrees (we'll do custom 0,90,180,270)
    translate: 0.1    # Translation
    scale: 0.5        # Scaling
    shear: 0.0        # Shear
    perspective: 0.0  # Perspective transform
    flipud: 0.25      # Vertical flip probability
    fliplr: 0.5       # Horizontal flip probability
    mosaic: 1.0       # Mosaic augmentation probability
    mixup: 0.0        # MixUp probability
  
  # Dynamic augmentations (curriculum learning - applied every N epochs)
  dynamic:
    enabled: true
    apply_every_n_epochs: 2
    
    # Small geometric transformations
    xy_shift_range: [-10, 10]  # pixels (custom implementation)
    rotation_range: [-5, 5]    # degrees
    
    # Image quality augmentations
    gaussian_blur_sigma: [0.5, 1.5]
    sharpen_alpha: [0.5, 1.5]
    
    # Intensity augmentations
    intensity_offset_range: [-15, 15]
    contrast_scale_range: [0.85, 1.15]
    gamma_range: [0.9, 1.1]
    
    # Additional recommended augmentations
    elastic_transform_enabled: true
    noise_enabled: true
    motion_blur_enabled: false  # Can interfere with defect detection
    cutout_enabled: true
    cutout_ratio: 0.1
  
  # Custom rotations (0, 90, 180, 270)
  custom_rotations:
    enabled: true
    angles: [0, 90, 180, 270]
    probability: 0.25  # Apply one of these rotations with this probability
  
  # Border handling
  border_mode: "reflect"  # mirror border
  
  # Random crop settings
  random_crop:
    enabled: true
    min_scale: 0.9  # Crop at least 90% of image
    max_scale: 1.0

# Training Settings
training:
  num_epochs: 50
  learning_rate: 0.01  # YOLO default, works well
  lr0: 0.01           # Initial learning rate
  lrf: 0.01           # Final learning rate (lr0 * lrf)
  momentum: 0.937
  weight_decay: 0.0005
  warmup_epochs: 3.0
  warmup_momentum: 0.8
  warmup_bias_lr: 0.1
  
  # Optimizer
  optimizer: "SGD"     # SGD works well for YOLO
  # Alternative: "AdamW"
  
  # Scheduler
  scheduler: "cosine"  # Cosine annealing
  
  # Checkpointing
  save_period: 5       # Save checkpoint every N epochs
  project: "runs/seg"
  name: "swir_board_inspection"
  
  # Early stopping
  patience: 10         # Stop if no improvement for N epochs
  
  # Mixed precision (not available on CPU)
  amp: false

# Advanced Settings
advanced:
  # Loss gains
  box_gain: 7.5
  cls_gain: 0.5
  dfl_gain: 1.5
  
  # Label smoothing
  label_smoothing: 0.0
  
  # Multi-scale training
  multi_scale: false   # Disable for CPU to save memory
  
  # Cache images
  cache: false         # Set to true if you have enough RAM

# Evaluation Settings
evaluation:
  metrics: ["map50", "map", "precision", "recall", "iou"]
  save_predictions: true
  visualization_samples: 10
  conf_threshold: 0.25  # Confidence threshold for predictions
  iou_threshold: 0.45   # IoU threshold for NMS

# Random Seed
seed: 42

# Device
device: "cpu"  # or "cuda", "mps", or leave empty for auto-detect
```

---

## 🚀 Training Script

Save as `scripts/train_yolo.py`:

```python
#!/usr/bin/env python3
"""
YOLOv8-Seg Training Script with Curriculum Learning for SWIR Board Inspection.
Supports CPU training with configurable augmentations.
"""

import os
import yaml
import random
import numpy as np
import torch
import cv2
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import shutil


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cv2.setRNGSeed(seed)


class CustomAugmentations:
    """Custom augmentations for curriculum learning."""
    
    def __init__(self, config):
        self.config = config
        self.aug_config = config['augmentation']
        
    def apply_dynamic_augmentation(self, image, mask=None, epoch=0):
        """Apply dynamic augmentations based on epoch."""
        
        dynamic_cfg = self.aug_config['dynamic']
        
        # Check if dynamic augmentations should be applied this epoch
        if not dynamic_cfg.get('enabled', True):
            return image, mask
        
        if epoch % dynamic_cfg.get('apply_every_n_epochs', 2) != 0:
            return image, mask
        
        # Build dynamic augmentation pipeline
        transforms = []
        
        # Small XY shift
        xy_shift = dynamic_cfg.get('xy_shift_range', [-10, 10])
        input_size = self.config['data']['input_size']
        transforms.append(A.Affine(
            translate_percent={"x": (xy_shift[0]/input_size, xy_shift[1]/input_size),
                              "y": (xy_shift[0]/input_size, xy_shift[1]/input_size)},
            p=0.5
        ))
        
        # Small rotation
        rot_range = dynamic_cfg.get('rotation_range', [-5, 5])
        transforms.append(A.Affine(rotate=rot_range, p=0.5))
        
        # Gaussian blur
        blur_sigma = dynamic_cfg.get('gaussian_blur_sigma', [0.5, 1.5])
        transforms.append(A.GaussianBlur(blur_limit=(3, 7), sigma_limit=blur_sigma, p=0.3))
        
        # Sharpening
        sharp_alpha = dynamic_cfg.get('sharpen_alpha', [0.5, 1.5])
        transforms.append(A.Sharpen(alpha=sharp_alpha, lightness=(0.8, 1.2), p=0.3))
        
        # Intensity offset
        intensity_range = dynamic_cfg.get('intensity_offset_range', [-15, 15])
        transforms.append(A.Lambda(
            name="intensity_offset",
            apply=lambda img, **params: self._apply_intensity_offset(img, intensity_range),
            p=0.4
        ))
        
        # Contrast stretch
        contrast_range = dynamic_cfg.get('contrast_scale_range', [0.85, 1.15])
        transforms.append(A.Lambda(
            name="contrast_stretch",
            apply=lambda img, **params: self._apply_contrast_stretch(img, contrast_range),
            p=0.4
        ))
        
        # Gamma correction
        gamma_range = dynamic_cfg.get('gamma_range', [0.9, 1.1])
        transforms.append(A.RandomGamma(gamma_limit=gamma_range, p=0.3))
        
        # Elastic transform
        if dynamic_cfg.get('elastic_transform_enabled', True):
            transforms.append(A.ElasticTransform(
                alpha=dynamic_cfg.get('elastic_alpha', 50),
                sigma=dynamic_cfg.get('elastic_sigma', 5),
                p=0.2
            ))
        
        # Noise
        if dynamic_cfg.get('noise_enabled', True):
            transforms.append(A.GaussNoise(var_limit=(5.0, 20.0), p=0.3))
        
        # Cutout
        if dynamic_cfg.get('cutout_enabled', True):
            transforms.append(A.CoarseDropout(
                max_holes=8,
                max_height=int(input_size * dynamic_cfg.get('cutout_ratio', 0.1)),
                max_width=int(input_size * dynamic_cfg.get('cutout_ratio', 0.1)),
                p=0.2
            ))
        
        # Compose transforms
        transform = A.Compose(transforms)
        
        if mask is not None:
            augmented = transform(image=image, mask=mask)
            return augmented['image'], augmented['mask']
        else:
            augmented = transform(image=image)
            return augmented['image'], None
    
    def _apply_intensity_offset(self, image, offset_range):
        """Apply random intensity offset."""
        offset = np.random.uniform(offset_range[0], offset_range[1])
        return np.clip(image.astype(np.float32) + offset, 0, 255).astype(np.uint8)
    
    def _apply_contrast_stretch(self, image, scale_range):
        """Apply random contrast scaling."""
        scale = np.random.uniform(scale_range[0], scale_range[1])
        mean_val = np.mean(image)
        return np.clip((image - mean_val) * scale + mean_val, 0, 255).astype(np.uint8)


def train(config_path, args):
    """Main training function."""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
        config['training']['lr0'] = args.learning_rate
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.input_size:
        config['data']['input_size'] = args.input_size
    if args.model:
        config['model']['variant'] = args.model
    
    # Set seed
    set_seed(config.get('seed', 42))
    
    # Create project directory
    project_dir = os.path.join(config['training']['project'], config['training']['name'])
    os.makedirs(project_dir, exist_ok=True)
    
    # Initialize YOLO model
    print(f"Loading YOLOv8 model: {config['model']['variant']}")
    
    # Check if model variant is a path or a preset
    if os.path.exists(config['model']['variant']):
        model = YOLO(config['model']['variant'])
    else:
        model = YOLO(config['model']['variant'])
    
    # Determine device
    device = config.get('device', '')
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Prepare training arguments
    train_args = {
        'data': config['data']['dataset_yaml'],
        'epochs': config['training']['num_epochs'],
        'batch': config['data']['batch_size'],
        'imgsz': config['data']['input_size'],
        'lr0': config['training']['lr0'],
        'lrf': config['training']['lrf'],
        'momentum': config['training']['momentum'],
        'weight_decay': config['training']['weight_decay'],
        'warmup_epochs': config['training']['warmup_epochs'],
        'optimizer': config['training']['optimizer'],
        'scheduler': config['training']['scheduler'],
        'patience': config['training']['patience'],
        'save_period': config['training']['save_period'],
        'project': config['training']['project'],
        'name': config['training']['name'],
        'amp': config['training']['amp'],
        'device': device,
        'seed': config.get('seed', 42),
        
        # Augmentation parameters (static)
        'hsv_h': config['augmentation']['static']['hsv_h'],
        'hsv_s': config['augmentation']['static']['hsv_s'],
        'hsv_v': config['augmentation']['static']['hsv_v'],
        'degrees': config['augmentation']['static']['degrees'],
        'translate': config['augmentation']['static']['translate'],
        'scale': config['augmentation']['static']['scale'],
        'shear': config['augmentation']['static']['shear'],
        'perspective': config['augmentation']['static']['perspective'],
        'flipud': config['augmentation']['static']['flipud'],
        'fliplr': config['augmentation']['static']['fliplr'],
        'mosaic': config['augmentation']['static']['mosaic'],
        'mixup': config['augmentation']['static']['mixup'],
        
        # Advanced settings
        'box': config['advanced']['box_gain'],
        'cls': config['advanced']['cls_gain'],
        'dfl': config['advanced']['dfl_gain'],
        'label_smoothing': config['advanced']['label_smoothing'],
        'multi_scale': config['advanced']['multi_scale'],
        'cache': config['advanced']['cache'],
    }
    
    print("\n" + "="*50)
    print("Starting YOLOv8-Seg Training")
    print("="*50)
    print(f"Model: {config['model']['variant']}")
    print(f"Dataset: {config['data']['dataset_yaml']}")
    print(f"Epochs: {config['training']['num_epochs']}")
    print(f"Batch Size: {config['data']['batch_size']}")
    print(f"Input Size: {config['data']['input_size']}")
    print(f"Learning Rate: {config['training']['lr0']}")
    print(f"Device: {device}")
    print("="*50 + "\n")
    
    # Train the model
    results = model.train(**train_args)
    
    print("\n🎉 Training completed!")
    print(f"Results saved to: {project_dir}")
    
    return model, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8-Seg for SWIR Board Inspection")
    parser.add_argument("--config", type=str, default="configs/yolo_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="Override learning rate")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Override batch size")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Override number of epochs")
    parser.add_argument("--input_size", type=int, default=None,
                       help="Override input size")
    parser.add_argument("--model", type=str, default=None,
                       help="Override model variant (e.g., yolov8n-seg.pt)")
    
    args = parser.parse_args()
    train(args.config, args)
```

---

## 📊 Evaluation and Visualization

Save as `scripts/evaluate_yolo.py`:

```python
#!/usr/bin/env python3
"""
Evaluation script for YOLOv8-Seg model with visualization.
"""

import os
import yaml
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
import argparse
import json
from sklearn.metrics import confusion_matrix
import seaborn as sns


def visualize_predictions(model, dataloader_or_images, config, num_samples=10, 
                         save_dir='evaluation_results', device='cpu'):
    """Visualize model predictions vs ground truth."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Get class names
    class_names = model.names
    colors = [
        [128, 0, 0],     # die - maroon
        [0, 128, 0],     # scratches - green
        [0, 0, 128],     # contamination - navy
        [128, 128, 0]    # other_defects - olive
    ]
    
    # Run validation
    print("Running validation...")
    metrics = model.val(data=config['data']['dataset_yaml'],
                       batch=1,
                       imgsz=config['data']['input_size'],
                       device=device,
                       plots=True,
                       save_dir=save_dir)
    
    print(f"\nValidation Metrics:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    try:
        # Confusion matrix is automatically saved by YOLO
        cm_path = os.path.join(save_dir, 'confusion_matrix.png')
        if os.path.exists(cm_path):
            print(f"Confusion matrix saved: {cm_path}")
    except Exception as e:
        print(f"Could not generate confusion matrix: {e}")
    
    # Visualize sample predictions
    print("\nGenerating prediction visualizations...")
    
    # Get sample images from validation set
    val_images_dir = os.path.join(
        os.path.dirname(config['data']['dataset_yaml']),
        'images', 'val'
    )
    
    if os.path.exists(val_images_dir):
        image_files = [f for f in os.listdir(val_images_dir) 
                      if f.endswith(('.png', '.jpg', '.jpeg'))][:num_samples]
        
        fig, axes = plt.subplots(num_samples, 2, figsize=(20, 5*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, img_file in enumerate(image_files):
            if i >= num_samples:
                break
            
            img_path = os.path.join(val_images_dir, img_file)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Run inference
            results = model.predict(source=img_path,
                                  imgsz=config['data']['input_size'],
                                  conf=config['evaluation']['conf_threshold'],
                                  iou=config['evaluation']['iou_threshold'],
                                  device=device)
            
            result = results[0]
            
            # Plot original image
            axes[i, 0].imshow(img_rgb)
            axes[i, 0].set_title(f"Original: {img_file}")
            axes[i, 0].axis('off')
            
            # Plot image with predictions
            pred_img = result.plot()
            axes[i, 1].imshow(pred_img)
            axes[i, 1].set_title("Predictions")
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        viz_path = os.path.join(save_dir, 'predictions_visualization.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Visualizations saved to: {viz_path}")


def evaluate(config_path, checkpoint_path, args):
    """Main evaluation function."""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    save_dir = 'evaluation_results'
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    device = config.get('device', '')
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        model = YOLO(checkpoint_path)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        # Load best model from training run
        project_dir = os.path.join(config['training']['project'], config['training']['name'])
        best_model_path = os.path.join(project_dir, 'weights', 'best.pt')
        
        if os.path.exists(best_model_path):
            model = YOLO(best_model_path)
            print(f"Loaded best model: {best_model_path}")
        else:
            print("No checkpoint found. Please train a model first.")
            return
    
    print(f"Using device: {device}")
    
    # Run validation
    print("\nRunning evaluation...")
    metrics = model.val(data=config['data']['dataset_yaml'],
                       batch=1,
                       imgsz=config['data']['input_size'],
                       device=device,
                       plots=True,
                       save_dir=save_dir)
    
    # Print metrics
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"mAP50 (IoU=0.50): {metrics.box.map50:.4f}")
    print(f"mAP50-95 (average): {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    
    # Per-class metrics
    print("\nPer-class Metrics:")
    print("-" * 50)
    print(f"{'Class':<20} {'mAP50':<10} {'mAP50-95':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 50)
    
    for i, class_name in enumerate(metrics.box.maps):
        map50 = metrics.box.maps[i]
        # Note: Detailed per-class metrics require parsing YOLO output
        print(f"{class_name:<20} {map50:.4f}")
    
    print("="*50)
    
    # Save metrics to file
    metrics_dict = {
        'map50': float(metrics.box.map50),
        'map': float(metrics.box.map),
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr),
        'per_class_map50': {name: float(metrics.box.maps[i]) 
                           for i, name in enumerate(metrics.box.maps)}
    }
    
    metrics_path = os.path.join(save_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"\n✅ Metrics saved to: {metrics_path}")
    
    # Generate visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        visualize_predictions(model, None, config,
                            num_samples=args.num_samples,
                            save_dir=save_dir,
                            device=device)
    
    print("\n✅ Evaluation complete!")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8-Seg model")
    parser.add_argument("--config", type=str, default="configs/yolo_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to model checkpoint")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualization plots")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of samples to visualize")
    
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args)
```

---

## 🏃 Running the Training

### Step 1: Prepare Your Data

```bash
# Organize your data
mkdir -p data/raw_images data/masks_coco data/splits data/yolo_dataset

# If you have CVAT exports in COCO format, convert them
python scripts/convert_coco_to_yolo.py \
    --coco_json data/masks_coco/annotations.json \
    --images_dir data/raw_images \
    --output_dir data/yolo_dataset \
    --classes die scratches contamination other_defects
```

### Step 2: Verify Dataset

Check that the conversion was successful:
```bash
# Count images and labels
echo "Training images:"
ls data/yolo_dataset/images/train | wc -l
echo "Training labels:"
ls data/yolo_dataset/labels/train | wc -l
echo "Validation images:"
ls data/yolo_dataset/images/val | wc -l
echo "Validation labels:"
ls data/yolo_dataset/labels/val | wc -l

# View dataset config
cat data/yolo_dataset/dataset.yaml
```

### Step 3: Start Training

```bash
# Activate environment
conda activate yolo_inspection

# Train with default config (recommended for CPU)
python scripts/train_yolo.py --config configs/yolo_config.yaml

# Or override parameters
python scripts/train_yolo.py \
    --config configs/yolo_config.yaml \
    --learning_rate 0.01 \
    --batch_size 2 \
    --epochs 50 \
    --input_size 640 \
    --model yolov8n-seg.pt

# Quick test with smaller model
python scripts/train_yolo.py \
    --config configs/yolo_config.yaml \
    --epochs 10 \
    --batch_size 1 \
    --model yolov8n-seg.pt
```

### Step 4: Evaluate the Model

```bash
# Evaluate best model
python scripts/evaluate_yolo.py \
    --config configs/yolo_config.yaml \
    --checkpoint runs/seg/swir_board_inspection/weights/best.pt \
    --visualize \
    --num_samples 15
```

---

## 📈 Monitoring Training

### Built-in YOLO Logging

YOLOv8 automatically logs:
- Training metrics (loss, mAP, precision, recall)
- Validation metrics
- Learning rate schedule
- GPU/CPU utilization

### Viewing Results

After training, check:
```bash
# Training curves and metrics
ls runs/seg/swir_board_inspection/

# View results.csv
cat runs/seg/swir_board_inspection/results.csv

# View confusion matrix
open runs/seg/swir_board_board_inspection/confusion_matrix.png

# View prediction examples
open runs/seg/swir_board_inspection/val_batch*_pred.jpg
```

### Key Metrics to Watch

1. **box_loss**: Bounding box regression loss - should decrease
2. **cls_loss**: Classification loss - should decrease
3. **dfl_loss**: Distribution focal loss - should decrease
4. **metrics/precision(B)**: Precision for boxes
5. **metrics/recall(B)**: Recall for boxes
6. **metrics/map50(B)**: mAP at IoU=0.50
7. **metrics/map50-95(B)**: Average mAP across IoU thresholds

---

## 🔧 Troubleshooting

### Common Issues

**1. Out of Memory on CPU**
```yaml
# In config, reduce:
data:
  batch_size: 1      # Minimum
  input_size: 320    # Smaller input
advanced:
  cache: false       # Don't cache images
  multi_scale: false # Disable multi-scale
```

**2. Slow Training on CPU**
- Use `yolov8n-seg.pt` (nano model)
- Reduce `input_size` to 320 or 416
- Set `workers: 0` in data settings
- Consider upgrading to GPU for production

**3. Poor Detection of Small Defects**
```yaml
# Adjust anchor boxes or use smaller model
model:
  variant: "yolov8s-seg.pt"  # Slightly larger model

# Increase resolution
data:
  input_size: 896  # Higher resolution

# Adjust loss weights
advanced:
  box_gain: 10.0   # More weight on box loss
  dfl_gain: 2.0
```

**4. Class Imbalance**
- Collect more data for rare classes
- Use weighted loss (modify training script)
- Oversample images with rare defects

**5. Overfitting**
```yaml
training:
  weight_decay: 0.001  # Increase regularization
  label_smoothing: 0.1 # Add label smoothing
  
augmentation:
  static:
    mosaic: 1.0        # Keep mosaic on
    mixup: 0.1         # Add some mixup
```

---

## 🎓 Tips for Best Results

1. **Start with Nano Model**: Begin with `yolov8n-seg.pt` for fast iteration
2. **Resolution Matters**: For small defects, use 640x640 or higher
3. **Data Quality**: Ensure accurate polygon annotations in CVAT
4. **Augmentation Balance**: YOLO has strong built-in augmentations
5. **Monitor Per-Class mAP**: Identify weak classes early
6. **Fine-tune Thresholds**: Adjust `conf_threshold` and `iou_threshold` for your use case
7. **Ensemble Models**: Combine predictions from multiple models for production

---

## 🚀 Deployment

### Export Model for Production

```bash
# Export to ONNX (for CPU inference)
python -c "from ultralytics import YOLO; model = YOLO('runs/seg/swir_board_inspection/weights/best.pt'); model.export(format='onnx', imgsz=640)"

# Export to TensorRT (for NVIDIA GPU)
python -c "from ultralytics import YOLO; model = YOLO('runs/seg/swir_board_inspection/weights/best.pt'); model.export(format='engine', imgsz=640)"

# Export to OpenVINO (for Intel CPU)
python -c "from ultralytics import YOLO; model = YOLO('runs/seg/swir_board_inspection/weights/best.pt'); model.export(format='openvino', imgsz=640)"
```

### Inference Example

```python
from ultralytics import YOLO

# Load model
model = YOLO('runs/seg/swir_board_inspection/weights/best.pt')

# Run inference
results = model.predict(
    source='test_image.png',
    imgsz=640,
    conf=0.25,
    iou=0.45
)

# Process results
for result in results:
    boxes = result.boxes      # Bounding boxes
    masks = result.masks      # Segmentation masks
    classes = result.boxes.cls  # Class IDs
    confidences = result.boxes.conf  # Confidence scores
    
    # Draw on image
    annotated = result.plot()
```

---

## 📚 Comparison: YOLOv8 vs SegFormer

| Aspect | YOLOv8-Seg | SegFormer |
|--------|-----------|-----------|
| **Speed (CPU)** | ⭐⭐⭐⭐ Fast | ⭐⭐ Slower |
| **Accuracy** | ⭐⭐⭐ Good | ⭐⭐⭐⭐ Excellent |
| **Small Objects** | ⭐⭐⭐ Good | ⭐⭐⭐⭐ Better |
| **Ease of Use** | ⭐⭐⭐⭐⭐ Very Easy | ⭐⭐⭐ Moderate |
| **Deployment** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐ Good |
| **Training Time** | Hours | Days (on CPU) |
| **Instance Segmentation** | ✅ Yes | ❌ No (semantic only) |

**Recommendation**: 
- Use **YOLOv8** for feasibility studies, rapid prototyping, and deployment
- Use **SegFormer** for maximum accuracy when time/compute allows

---

## 📚 Next Steps

After mastering this pipeline:
1. Experiment with different YOLO variants (n, s, m, l, x)
2. Try ensemble methods
3. Implement tracking for video inspection
4. Deploy with ONNX Runtime or TensorRT
5. Explore active learning for efficient labeling

Good luck with your SWIR board inspection project! 🚀
