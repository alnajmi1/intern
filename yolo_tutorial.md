# YOLOv8-Seg Training Tutorial for SWIR Vitrox Board Inspection

## 📋 Overview
This tutorial guides you through training a **YOLOv8-Seg** (instance segmentation) model for inspecting SWIR Vitrox boards. We'll cover data preparation, augmentation strategies, training, and evaluation using the Ultralytics framework.

## 🎯 Target Classes
- **Background**: Non-defective areas (implicit)
- **Die**: The main chip/component
- **Scratches**: Linear defects
- **Contamination**: Particle/foreign material defects
- **Others**: Add as needed

---

## 📦 1. Environment Setup

### Prerequisites
- Python 3.8+
- CPU environment (training will be slower but functional)
- Conda package manager

### Step 1: Create Conda Environment
```bash
conda create -n swir_yolo python=3.10 -y
conda activate swir_yolo
```

### Step 2: Install Dependencies
```bash
# Core deep learning (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Ultralytics YOLOv8
pip install ultralytics albumentations opencv-python-headless

# Data handling and visualization
pip install pillow matplotlib tqdm scikit-learn

# COCO format tools
pip install pycocotools
```

### Step 3: Verify Installation
```python
import torch
import ultralytics
import albumentations as A

print(f"PyTorch version: {torch.__version__}")
print(f"Ultralytics version: {ultralytics.__version__}")
print(f"Albumentations version: {A.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Check YOLO installation
from ultralytics import YOLO
print("✅ YOLOv8 installed successfully!")
```

---

## 📁 2. Data Preparation

### Directory Structure (YOLO Format)
Organize your data as follows:
```
data/
├── raw_images/           # Original images from camera
├── cvat_annotations/     # Exported from CVAT (COCO JSON format)
├── yolo_dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── dataset.yaml          # YOLO dataset configuration
└── class_names.txt       # Class names list
```

### Step 1: Label Data with CVAT
1. Go to [cvat.ai](https://www.cvat.ai/) or self-hosted CVAT
2. Create a new project → Upload your SWIR images
3. Define classes: `die`, `scratches`, `contamination` (background is implicit)
4. Annotate using **polygon tools** for precise boundaries
5. Export dataset in **COCO JSON** format

### Step 2: Convert COCO to YOLO Format
Use the provided conversion script:

```bash
python scripts/convert_coco_to_yolo.py \
    --coco_json data/cvat_annotations/annotations.json \
    --images_dir data/raw_images \
    --output_dir data/yolo_dataset \
    --class_names "die,scratches,contamination" \
    --split_ratio 0.8
```

This creates:
- Images copied to `train/` and `val/` folders
- YOLO-format `.txt` label files (one per image)
- `dataset.yaml` configuration file

### YOLO Label Format
Each `.txt` file contains one line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```
All values are normalized (0-1).

For segmentation, polygons are represented as:
```
<class_id> x1 y1 x2 y2 x3 y3 ... xn yn
```

### Step 3: Create Dataset Configuration
The conversion script auto-generates `data/yolo_dataset/dataset.yaml`:

```yaml
# YOLOv8 Dataset Configuration
path: /full/path/to/data/yolo_dataset
train: images/train
val: images/val

# Classes
names:
  0: die
  1: scratches
  2: contamination

nc: 3  # number of classes
```

Update the `path` to absolute path if needed.

---

## ⚙️ 3. Configuration

### Training Configuration File: `configs/yolo_config.yaml`

```yaml
# Model Configuration
model:
  name: "yolov8n-seg.pt"  # Nano version for CPU (or s/m/l/x for larger)
  input_shape: 640  # Must be multiple of 32
  
# Data Configuration
data:
  dataset_yaml: "data/yolo_dataset/dataset.yaml"
  
# Augmentation Configuration (Custom Curriculum Learning)
augmentation:
  # Built-in YOLO augmentations (always on)
  hsv_h: 0.015        # Hue augmentation
  hsv_s: 0.7          # Saturation augmentation
  hsv_v: 0.4          # Value augmentation
  degrees: 0.0        # Rotation (we'll do custom 90° increments)
  translate: 0.1      # Translation
  scale: 0.5          # Scaling
  shear: 0.0          # Shear
  perspective: 0.0    # Perspective
  flipud: 0.0         # Flip up-down (we'll do custom)
  fliplr: 0.5         # Flip left-right
  mosaic: 1.0         # Mosaic augmentation
  mixup: 0.0          # MixUp augmentation
  copy_paste: 0.0     # Copy-paste for segmentation
  
  # Custom curriculum learning (applied every N epochs via callback)
  curriculum:
    enabled: true
    interval: 2  # Apply every N epochs
    small_shift_range: 10  # pixels
    small_rotation_range: 15  # degrees
    gaussian_blur_kernel: [3, 5]
    gaussian_blur_sigma: [0.5, 2.0]
    unsharp_mask_strength: [0.2, 0.5]
    intensity_offset: [-20, 20]
    contrast_scale: [0.8, 1.2]
    border_mode: "mirror"
    
# Training Configuration
training:
  batch_size: 4  # Reduce for CPU
  num_epochs: 50
  learning_rate: 0.01  # YOLO default is higher
  optimizer: "SGD"  # or "AdamW"
  momentum: 0.937
  weight_decay: 0.0005
  warmup_epochs: 3
  warmup_momentum: 0.8
  warmup_bias_lr: 0.1
  
  # Scheduler
  scheduler: "cosine"  # or "linear", "step"
  
  # Early stopping
  patience: 20  # Stop if no improvement
  
  # Mixed precision (disable for CPU)
  amp: false
  
# Hardware
hardware:
  device: "cpu"
  workers: 2  # Data loader workers
  
# Evaluation
evaluation:
  save_predictions: true
  visualize_every: 5  # Visualize predictions every N epochs
  metrics: ["mAP50", "mAP50-95", "segmentation_iou"]
```

---

## 🔄 4. Augmentation Strategy

### Built-in YOLOv8 Augmentations
YOLOv8 includes powerful built-in augmentations:
- **Mosaic**: Combines 4 images (excellent for small objects)
- **MixUp**: Blends two images
- **HSV Augmentation**: Color jittering
- **Geometric Transforms**: Rotation, translation, scale
- **Flips**: Horizontal/vertical
- **Copy-Paste**: For segmentation tasks

### Custom Curriculum Learning Augmentations
We add additional augmentations every N epochs:

#### Static Augmentations (Every Batch)
- **Rotations**: 0°, 90°, 180°, 270° (via custom logic)
- **Flips**: X, Y, XY combinations
- **Random Crop**: Minimal distortion (original - 32px)
- **Mirror Border**: Prevents edge artifacts

#### Dynamic Augmentations (Every N Epochs)
- **Small XY Shift**: ±10 pixels
- **Small Rotation**: ±15°
- **Gaussian Blur**: Kernel 3-5, σ 0.5-2.0
- **Sharpening**: Unsharp mask strength 0.2-0.5
- **Intensity Offset**: ±20 grayscale values
- **Contrast Stretch**: 0.8x to 1.2x scaling

### AI-Recommended Additional Augmentations
- **Elastic Transformations**: Simulate physical warping
- **Defocus Blur**: Optical imperfections
- **JPEG Compression Artifacts**: Real-world degradation
- **Poisson Noise**: Sensor noise simulation
- **Grid Mask**: Occlusion simulation

---

## 🚀 5. Training Script

### Main Training Script: `train_yolo.py`

```python
#!/usr/bin/env python3
"""
YOLOv8-Seg Training Script for SWIR Vitrox Board Inspection
Supports curriculum learning with progressive augmentations
"""

import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List
import json

import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.engine.trainer import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG
import matplotlib.pyplot as plt


class CustomAugmenter:
    """Custom augmentation pipeline for curriculum learning"""
    
    def __init__(self, config: dict):
        self.config = config['augmentation']['curriculum']
        self.input_shape = config['model']['input_shape']
        
        self.curriculum_transforms = A.Compose([
            A.ShiftScaleRotate(
                shift_limit=self.config.get('small_shift_range', 10) / self.input_shape,
                rotate_limit=self.config.get('small_rotation_range', 15),
                scale_limit=0.05,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.8
            ),
            
            A.OneOf([
                A.GaussianBlur(
                    blur_limit=tuple(self.config.get('gaussian_blur_kernel', [3, 5])),
                    sigma_limit=tuple(self.config.get('gaussian_blur_sigma', [0.5, 2.0])),
                    p=1
                ),
                A.MotionBlur(blur_limit=3, p=1),
                A.Defocus(blur_limit=(3, 5), p=1),
            ], p=0.5),
            
            A.UnsharpMask(
                radius=3,
                alpha=tuple(self.config.get('unsharp_mask_strength', [0.2, 0.5])),
                threshold=10,
                p=0.5
            ),
            
            A.RandomBrightnessContrast(
                brightness_limit=tuple(self.config.get('intensity_offset', [-20, 20])) / 255.0,
                contrast_limit=tuple(self.config.get('contrast_scale', [0.8, 1.2])),
                p=0.7
            ),
            
            # Additional augmentations
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=0,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.3
            ),
            
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
            
            # JPEG compression artifacts
            A.ImageCompression(quality_lower=75, quality_upper=100, p=0.3),
            
        ], is_check_shapes=False)
    
    def apply(self, image: np.ndarray, masks: np.ndarray = None) -> tuple:
        """Apply curriculum augmentations"""
        if masks is not None:
            transformed = self.curriculum_transforms(image=image, masks=masks)
            return transformed['image'], transformed['masks']
        else:
            transformed = self.curriculum_transforms(image=image)
            return transformed['image'], None


class CustomYOLOTrainer(DetectionTrainer):
    """Custom YOLO trainer with curriculum learning support"""
    
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.epoch = 0
        self.custom_augmenter = None
        
        if 'config_path' in overrides:
            with open(overrides['config_path'], 'r') as f:
                self.config = yaml.safe_load(f)
            
            if self.config['augmentation']['curriculum']['enabled']:
                self.custom_augmenter = CustomAugmenter(self.config)
    
    def get_dataset(self):
        """Get dataset with custom augmentations"""
        dataset = super().get_dataset()
        
        # Store original __getitem__
        original_getitem = dataset.__getitem__
        
        def custom_getitem(index):
            # Get original item
            item = original_getitem(index)
            
            # Apply curriculum augmentations if epoch matches
            if (self.custom_augmenter and 
                self.epoch % self.config['augmentation']['curriculum']['interval'] == 0 and
                self.epoch > 0):
                
                # Apply custom augmentations to image
                img = item['img'].numpy().transpose(1, 2, 0).copy()
                img_aug, _ = self.custom_augmenter.apply(img)
                item['img'] = torch.from_numpy(img_aug.transpose(2, 0, 1)).float()
                
                print(f"  📈 Applied curriculum augmentations at epoch {self.epoch}")
            
            return item
        
        dataset.__getitem__ = custom_getitem
        return dataset
    
    def _setup_train(self, world_size):
        """Setup training with epoch tracking"""
        super()._setup_train(world_size)
        self.epoch = 0
    
    def _do_train(self, world_size=1):
        """Custom training loop with epoch tracking"""
        for epoch in range(self.args.epochs):
            self.epoch = epoch
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.args.epochs}")
            
            if (self.custom_augmenter and 
                epoch % self.config['augmentation']['curriculum']['interval'] == 0 and
                epoch > 0):
                print("🔄 Applying curriculum learning augmentations this epoch!")
            
            super()._do_train(world_size)


def train(config_path: str):
    """Main training function"""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("YOLOv8-Seg Training for SWIR Vitrox Board Inspection")
    print("="*60)
    
    # Load pretrained model
    model_name = config['model']['name']
    print(f"\n📦 Loading model: {model_name}")
    model = YOLO(model_name)
    
    # Prepare training arguments
    train_args = {
        'data': config['data']['dataset_yaml'],
        'epochs': config['training']['num_epochs'],
        'batch': config['training']['batch_size'],
        'imgsz': config['model']['input_shape'],
        'lr0': config['training']['learning_rate'],
        'optimizer': config['training']['optimizer'],
        'momentum': config['training']['momentum'],
        'weight_decay': config['training']['weight_decay'],
        'warmup_epochs': config['training']['warmup_epochs'],
        'warmup_momentum': config['training']['warmup_momentum'],
        'warmup_bias_lr': config['training']['warmup_bias_lr'],
        'scheduler': config['training']['scheduler'],
        'patience': config['training']['patience'],
        'amp': config['training']['amp'],
        'device': config['hardware']['device'],
        'workers': config['hardware']['workers'],
        
        # Augmentation parameters
        'hsv_h': config['augmentation']['hsv_h'],
        'hsv_s': config['augmentation']['hsv_s'],
        'hsv_v': config['augmentation']['hsv_v'],
        'degrees': config['augmentation']['degrees'],
        'translate': config['augmentation']['translate'],
        'scale': config['augmentation']['scale'],
        'shear': config['augmentation']['shear'],
        'perspective': config['augmentation']['perspective'],
        'flipud': config['augmentation']['flipud'],
        'fliplr': config['augmentation']['fliplr'],
        'mosaic': config['augmentation']['mosaic'],
        'mixup': config['augmentation']['mixup'],
        'copy_paste': config['augmentation']['copy_paste'],
        
        # Custom config for curriculum learning
        'config_path': config_path,
        
        # Project settings
        'project': 'runs/seg',
        'name': 'swir_vitrox_yolo',
        'exist_ok': True,
        'verbose': True,
        'save': True,
        'save_period': config['evaluation']['visualize_every'],
    }
    
    print(f"\n⚙️ Training Configuration:")
    print(f"  - Model: {model_name}")
    print(f"  - Input size: {config['model']['input_shape']}x{config['model']['input_shape']}")
    print(f"  - Batch size: {config['training']['batch_size']}")
    print(f"  - Epochs: {config['training']['num_epochs']}")
    print(f"  - Learning rate: {config['training']['learning_rate']}")
    print(f"  - Device: {config['hardware']['device']}")
    print(f"  - Curriculum learning: {'Enabled' if config['augmentation']['curriculum']['enabled'] else 'Disabled'}")
    
    # Start training
    print("\n🚀 Starting training...")
    results = model.train(**train_args)
    
    # Save training curves
    plot_training_curves(results)
    
    print(f"\n🎉 Training completed!")
    print(f"💾 Best model saved to: runs/seg/swir_vitrox_yolo/weights/best.pt")
    
    return model, results


def plot_training_curves(results):
    """Plot training curves from YOLO results"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot loss curves
        if hasattr(results, 'results_dict'):
            epochs = list(range(1, len(results.results_dict['train/box_loss']) + 1))
            
            axes[0, 0].plot(epochs, results.results_dict['train/box_loss'], label='Box Loss', marker='o')
            axes[0, 0].plot(epochs, results.results_dict['train/cls_loss'], label='Cls Loss', marker='s')
            axes[0, 0].plot(epochs, results.results_dict['train/dfl_loss'], label='DFL Loss', marker='^')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training Losses')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            axes[0, 1].plot(epochs, results.results_dict['metrics/val/mAP50-95(B)'], label='mAP50-95(Box)', marker='o')
            axes[0, 1].plot(epochs, results.results_dict['metrics/val/mAP50-95(M)'], label='mAP50-95(Mask)', marker='s')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('mAP')
            axes[0, 1].set_title('Validation mAP')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            axes[1, 0].plot(epochs, results.results_dict['metrics/val/precision'], label='Precision', marker='o')
            axes[1, 0].plot(epochs, results.results_dict['metrics/val/recall'], label='Recall', marker='s')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].set_title('Precision & Recall')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Placeholder for 4th plot
        axes[1, 1].text(0.5, 0.5, 'Training Complete!', 
                       ha='center', va='center', fontsize=16)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('training_curves_yolo.png', dpi=300)
        plt.show()
        print("📊 Training curves saved to 'training_curves_yolo.png'")
        
    except Exception as e:
        print(f"⚠️ Could not plot curves: {e}")


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8-Seg for SWIR inspection')
    parser.add_argument('--config', type=str, default='configs/yolo_config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    model, results = train(args.config)


if __name__ == "__main__":
    main()
```

---

## 🏃 6. Running Training

### Step 1: Prepare Your Data
```bash
# Organize directories
mkdir -p data/yolo_dataset/{images,labels}/{train,val}

# Convert COCO (from CVAT) to YOLO format
python scripts/convert_coco_to_yolo.py \
    --coco_json data/cvat_annotations/annotations.json \
    --images_dir data/raw_images \
    --output_dir data/yolo_dataset \
    --class_names "die,scratches,contamination" \
    --split_ratio 0.8
```

### Step 2: Update Dataset YAML
Edit `data/yolo_dataset/dataset.yaml` to use absolute path:
```yaml
path: /absolute/path/to/workspace/data/yolo_dataset
train: images/train
val: images/val

names:
  0: die
  1: scratches
  2: contamination

nc: 3
```

### Step 3: Start Training
```bash
python train_yolo.py --config configs/yolo_config.yaml
```

### Alternative: Direct YOLO Command
You can also train directly with YOLO CLI:
```bash
yolo segment train \
    model=yolov8n-seg.pt \
    data=data/yolo_dataset/dataset.yaml \
    epochs=50 \
    batch=4 \
    imgsz=640 \
    lr0=0.01 \
    device=cpu \
    project=runs/seg \
    name=swir_vitrox_direct
```

### Expected Output
```
============================================================
YOLOv8-Seg Training for SWIR Vitrox Board Inspection
============================================================

📦 Loading model: yolov8n-seg.pt

⚙️ Training Configuration:
  - Model: yolov8n-seg.pt
  - Input size: 640x640
  - Batch size: 4
  - Epochs: 50
  - Learning rate: 0.01
  - Device: cpu
  - Curriculum learning: Enabled

🚀 Starting training...

Ultralytics YOLOv8.0.196 🚀 Python-3.10.12 torch-2.0.1+cpu CPU
engine/trainer: task=seg, mode=train, model=yolov8n-seg.pt
Dataset 'data/yolo_dataset/dataset.yaml' images not found, using auto-download...
Downloading dataset...

Epoch   GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
  0%|          | 0/20 [00:00<?, ?it/s]
      1        0GB     0.8234     0.4521     1.2345         15        640: 100%|██████████| 20/20 [00:45<00:00,  2.28s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95)
                   all         39        156      0.721      0.654      0.687      0.421      0.698      0.623      0.654      0.398

📈 Applied curriculum augmentations at epoch 2
...
🎉 Training completed!
💾 Best model saved to: runs/seg/swir_vitrox_yolo/weights/best.pt
```

---

## 📊 7. Model Evaluation

### Inference Script: `evaluate_yolo.py`

```python
#!/usr/bin/env python3
"""
Evaluation script for trained YOLOv8-Seg model
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
import json

from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


class YOLOEvaluator:
    def __init__(self, model_path: str, class_names: list = None):
        self.device = 'cpu'
        
        # Load model
        print(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # Class names
        if class_names is None:
            self.class_names = ['die', 'scratches', 'contamination']
        else:
            self.class_names = class_names
        
        self.num_classes = len(self.class_names)
    
    def predict(self, image_path: str, conf_threshold: float = 0.25) -> dict:
        """Run inference on single image"""
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            device=self.device,
            verbose=False
        )
        
        result = results[0]
        
        # Extract predictions
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
        scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else []
        classes = result.boxes.cls.cpu().numpy() if result.boxes is not None else []
        masks = result.masks.data.cpu().numpy() if result.masks is not None else []
        
        return {
            'boxes': boxes,
            'scores': scores,
            'classes': classes,
            'masks': masks
        }
    
    def evaluate_dataset(self, image_dir: str, label_dir: str, conf_threshold: float = 0.25):
        """Evaluate on dataset with ground truth"""
        image_paths = sorted(list(Path(image_dir).glob("*.png")) + 
                            list(Path(image_dir).glob("*.jpg")))
        
        all_preds = []
        all_trues = []
        ious_per_class = [[] for _ in range(self.num_classes)]
        
        print(f"Evaluating {len(image_paths)} images...")
        
        for img_path in image_paths:
            # Get predictions
            pred = self.predict(str(img_path), conf_threshold)
            
            # Load ground truth
            label_path = Path(label_dir) / (img_path.stem + ".txt")
            true_objects = []
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = list(map(float, line.strip().split()))
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            # For segmentation, remaining values are polygon points
                            true_objects.append({
                                'class': class_id,
                                'polygon': parts[1:] if len(parts) > 5 else None
                            })
            
            # Collect predictions
            for cls, score in zip(pred['classes'], pred['scores']):
                all_preds.append(int(cls))
            
            # Collect ground truth
            for obj in true_objects:
                all_trues.append(obj['class'])
        
        # Classification report
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        if len(all_preds) > 0 and len(all_trues) > 0:
            print(classification_report(
                all_trues, all_preds,
                target_names=self.class_names,
                digits=4,
                zero_division=0
            ))
        else:
            print("No predictions or ground truth found!")
        
        print(f"\nTotal predictions: {len(all_preds)}")
        print(f"Total ground truth: {len(all_trues)}")
    
    def visualize_prediction(self, image_path: str, output_path: str = None, 
                            conf_threshold: float = 0.25):
        """Visualize prediction on image"""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            device=self.device,
            verbose=False
        )
        
        result = results[0]
        
        # Plot using YOLO's built-in visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(image_rgb)
        
        # Draw masks
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            
            colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
            
            for i, (mask, box, cls, score) in enumerate(zip(masks, boxes, classes, scores)):
                color = colors[i]
                
                # Draw mask
                mask_img = np.zeros((*mask.shape, 4), dtype=np.uint8)
                mask_img[mask > 0.5] = [*color[:3], 0.5]  # RGBA with 50% opacity
                
                ax.imshow(mask_img, alpha=0.5)
                
                # Draw bounding box
                rect = plt.Rectangle(
                    (box[0], box[1]),
                    box[2] - box[0],
                    box[3] - box[1],
                    fill=False,
                    color=color[:3],
                    linewidth=2,
                    label=f'{self.class_names[int(cls)]}: {score:.2f}'
                )
                ax.add_patch(rect)
        
        ax.set_title(f'Prediction: {Path(image_path).name}')
        ax.axis('off')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
        
        plt.show()
    
    def batch_visualize(self, image_dir: str, output_dir: str, 
                       num_images: int = 10, conf_threshold: float = 0.25):
        """Visualize predictions for multiple images"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        image_paths = sorted(list(Path(image_dir).glob("*.png")) + 
                            list(Path(image_dir).glob("*.jpg")))[:num_images]
        
        print(f"Visualizing {len(image_paths)} images...")
        
        for img_path in image_paths:
            output_path = Path(output_dir) / f"pred_{img_path.name}"
            self.visualize_prediction(str(img_path), str(output_path), conf_threshold)


def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLOv8-Seg model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pt file)')
    parser.add_argument('--images', type=str, default=None,
                       help='Path to test images directory')
    parser.add_argument('--labels', type=str, default=None,
                       help='Path to test labels directory (for evaluation)')
    parser.add_argument('--visualize', type=str, default=None,
                       help='Path to single image for visualization')
    parser.add_argument('--batch_visualize', type=str, default=None,
                       help='Directory for batch visualization')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--class_names', type=str, nargs='+', 
                       default=['die', 'scratches', 'contamination'],
                       help='Class names')
    
    args = parser.parse_args()
    
    evaluator = YOLOEvaluator(args.model, args.class_names)
    
    if args.visualize:
        evaluator.visualize_prediction(args.visualize, conf_threshold=args.conf)
    elif args.batch_visualize:
        evaluator.batch_visualize(args.batch_visualize, args.output_dir, 
                                 num_images=10, conf_threshold=args.conf)
    elif args.images:
        evaluator.evaluate_dataset(args.images, args.labels or args.images, 
                                  conf_threshold=args.conf)
    else:
        print("Please specify --visualize, --batch_visualize, or --images")


if __name__ == "__main__":
    main()
```

### Run Evaluation
```bash
# Full dataset evaluation
python evaluate_yolo.py \
    --model runs/seg/swir_vitrox_yolo/weights/best.pt \
    --images data/yolo_dataset/images/val \
    --labels data/yolo_dataset/labels/val \
    --conf 0.25

# Visualize single prediction
python evaluate_yolo.py \
    --model runs/seg/swir_vitrox_yolo/weights/best.pt \
    --visualize data/yolo_dataset/images/val/sample_001.png \
    --conf 0.25

# Batch visualization
python evaluate_yolo.py \
    --model runs/seg/swir_vitrox_yolo/weights/best.pt \
    --batch_visualize data/yolo_dataset/images/val \
    --output_dir visualizations \
    --conf 0.25
```

---

## 🛠️ 8. Helper Scripts

### COCO to YOLO Converter: `scripts/convert_coco_to_yolo.py`

```python
#!/usr/bin/env python3
"""
Convert CVAT COCO JSON annotations to YOLO format
"""

import json
import argparse
import shutil
from pathlib import Path
import numpy as np
from PIL import Image


def coco_to_yolo(coco_json: str, images_dir: str, output_dir: str,
                 class_names: list, split_ratio: float = 0.8):
    """Convert COCO annotations to YOLO format"""
    
    # Load COCO annotations
    with open(coco_json, 'r') as f:
        coco_data = json.load(f)
    
    # Create class name to ID mapping
    name_to_id = {name: idx for idx, name in enumerate(class_names)}
    
    # Get all images and annotations
    images = {img['id']: img for img in coco_data['images']}
    annotations = coco_data['annotations']
    
    # Group annotations by image
    img_annotations = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)
    
    # Split into train/val
    image_ids = list(images.keys())
    np.random.seed(42)
    np.random.shuffle(image_ids)
    split_idx = int(len(image_ids) * split_ratio)
    train_ids = image_ids[:split_idx]
    val_ids = image_ids[split_idx:]
    
    # Create output directories
    for split in ['train', 'val']:
        Path(f"{output_dir}/images/{split}").mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/labels/{split}").mkdir(parents=True, exist_ok=True)
    
    def process_image(img_id: int, split: str):
        img_info = images[img_id]
        img_name = img_info['file_name']
        width = img_info['width']
        height = img_info['height']
        
        # Copy image
        src_path = Path(images_dir) / img_name
        dst_path = Path(f"{output_dir}/images/{split}/{img_name}")
        
        if not src_path.exists():
            print(f"Warning: Image not found: {src_path}")
            return
        
        shutil.copy(src_path, dst_path)
        
        # Create YOLO label file
        label_path = Path(f"{output_dir}/labels/{split}/{Path(img_name).stem}.txt")
        
        lines = []
        if img_id in img_annotations:
            for ann in img_annotations[img_id]:
                category_id = ann['category_id']
                
                # Find class name
                cat_name = None
                for cat in coco_data['categories']:
                    if cat['id'] == category_id:
                        cat_name = cat['name']
                        break
                
                if cat_name and cat_name in name_to_id:
                    class_id = name_to_id[cat_name]
                    
                    # Get segmentation
                    segmentation = ann['segmentation']
                    if isinstance(segmentation, list):
                        for poly in segmentation:
                            # Convert to normalized coordinates
                            poly_normalized = []
                            for i in range(0, len(poly), 2):
                                x = poly[i] / width
                                y = poly[i + 1] / height
                                poly_normalized.extend([x, y])
                            
                            # YOLO format: class x1 y1 x2 y2 ... xn yn
                            line = f"{class_id} " + " ".join(map(str, poly_normalized))
                            lines.append(line)
        
        # Write label file
        with open(label_path, 'w') as f:
            f.write("\n".join(lines))
        
        print(f"Processed: {img_name} ({split}) - {len(lines)} objects")
    
    # Process train and val sets
    for img_id in train_ids:
        process_image(img_id, 'train')
    
    for img_id in val_ids:
        process_image(img_id, 'val')
    
    # Create dataset.yaml
    dataset_yaml = f"""# YOLOv8 Dataset Configuration
path: {Path(output_dir).resolve()}
train: images/train
val: images/val

# Classes
names:
"""
    for idx, name in enumerate(class_names):
        dataset_yaml += f"  {idx}: {name}\n"
    
    dataset_yaml += f"\nnc: {len(class_names)}\n"
    
    with open(f"{output_dir}/dataset.yaml", 'w') as f:
        f.write(dataset_yaml)
    
    print(f"\n✅ Conversion complete!")
    print(f"Train: {len(train_ids)} images")
    print(f"Val: {len(val_ids)} images")
    print(f"Dataset config: {output_dir}/dataset.yaml")


def main():
    parser = argparse.ArgumentParser(description='Convert COCO to YOLO format')
    parser.add_argument('--coco_json', type=str, required=True,
                       help='Path to COCO JSON file from CVAT')
    parser.add_argument('--images_dir', type=str, required=True,
                       help='Directory containing original images')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for YOLO dataset')
    parser.add_argument('--class_names', type=str, required=True,
                       help='Comma-separated class names (e.g., "die,scratches,contamination")')
    parser.add_argument('--split_ratio', type=float, default=0.8,
                       help='Train/val split ratio (default: 0.8)')
    
    args = parser.parse_args()
    
    class_names = [name.strip() for name in args.class_names.split(',')]
    
    coco_to_yolo(
        args.coco_json,
        args.images_dir,
        args.output_dir,
        class_names,
        args.split_ratio
    )


if __name__ == "__main__":
    main()
```

---

## 📈 9. Tips for Better Results

### Model Selection
- **yolov8n-seg.pt**: Fastest, smallest (best for CPU)
- **yolov8s-seg.pt**: Good balance
- **yolov8m/l/x-seg.pt**: More accurate but slower

### Hyperparameter Tuning
- **Learning Rate**: 0.01 is YOLO default; try 0.001 for fine-tuning
- **Batch Size**: Maximize based on available memory
- **Image Size**: 640 is standard; increase for small defects

### Augmentation Strategy
- Start with default YOLO augmentations
- Enable curriculum learning after baseline
- Adjust mosaic/mixup based on defect sizes

### Handling Class Imbalance
- Use `class_weights` in training
- Oversample rare classes
- Adjust confidence thresholds per class

---

## 🔧 10. Troubleshooting

### Common Issues

**Issue**: Training very slow on CPU
- **Solution**: Use yolov8n-seg.pt, reduce imgsz to 416, decrease batch_size

**Issue**: Poor detection of small scratches
- **Solution**: Increase imgsz to 1280, enable high-res training, adjust anchor boxes

**Issue**: Model overfitting
- **Solution**: Increase augmentations, enable dropout, reduce model size

**Issue**: CUDA out of memory (if using GPU later)
- **Solution**: Reduce batch_size, imgsz, or use gradient accumulation

---

## 📚 11. Next Steps

1. **Label data** in CVAT with polygon annotations
2. **Convert to YOLO format** using conversion script
3. **Start baseline training** with default augmentations
4. **Enable curriculum learning** for improved robustness
5. **Evaluate and tune** hyperparameters
6. **Deploy model** for production inference

---

## 🆚 SegFormer vs YOLO Comparison

| Feature | SegFormer | YOLOv8-Seg |
|---------|-----------|------------|
| **Architecture** | Transformer-based | CNN-based |
| **Speed (CPU)** | Slower | Faster |
| **Accuracy** | Higher mIoU | Good mAP |
| **Memory** | Higher | Lower |
| **Ease of Use** | More setup | Very easy |
| **Best For** | Precise boundaries | Real-time inference |

---

## 📞 Support

For questions about:
- **CVAT labeling**: Request separate tutorial
- **Data conversion**: Check script outputs
- **Training issues**: Review logs and curves
- **Deployment**: Contact for optimization guide

Good luck with your SWIR Vitrox board inspection project! 🚀
