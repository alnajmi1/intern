# SegFormer Training Tutorial for SWIR Vitrox Board Inspection

## 📋 Overview
This tutorial guides you through training a **SegFormer** segmentation model for inspecting SWIR Vitrox boards. We'll cover data preparation, augmentation strategies, training, and evaluation.

## 🎯 Target Classes
- **Background**: Non-defective areas
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
conda create -n swir_seg python=3.10 -y
conda activate swir_seg
```

### Step 2: Install Dependencies
```bash
# Core deep learning
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Hugging Face Transformers for SegFormer
pip install transformers albumentations opencv-python-headless

# Data handling and visualization
pip install pillow matplotlib tqdm scikit-learn

# COCO format support (for conversion)
pip install pycocotools
```

### Step 3: Verify Installation
```python
import torch
import transformers
import albumentations as A

print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"Albumentations version: {A.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

---

## 📁 2. Data Preparation

### Directory Structure
Organize your data as follows:
```
data/
├── raw_images/           # Original images from camera
├── cvat_annotations/     # Exported from CVAT (COCO JSON format)
├── processed/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── masks/
│       ├── train/
│       └── val/
└── class_mapping.json    # Class name to ID mapping
```

### Step 1: Label Data with CVAT
1. Go to [cvat.ai](https://www.cvat.ai/) or self-hosted CVAT
2. Create a new project → Upload your SWIR images
3. Define classes: `background`, `die`, `scratches`, `contamination`
4. Annotate using polygon tools for precise boundaries
5. Export dataset in **COCO JSON** format

### Step 2: Convert COCO to Indexed Masks
Use the provided conversion script (see `scripts/convert_coco_to_masks.py`):

```bash
python scripts/convert_coco_to_masks.py \
    --coco_json data/cvat_annotations/annotations.json \
    --images_dir data/raw_images \
    --output_dir data/processed \
    --class_mapping '{"background": 0, "die": 1, "scratches": 2, "contamination": 3}'
```

This creates indexed PNG masks where each pixel value corresponds to a class ID.

### Step 3: Create Class Mapping File
Save `data/class_mapping.json`:
```json
{
  "0": "background",
  "1": "die",
  "2": "scratches",
  "3": "contamination"
}
```

---

## ⚙️ 3. Configuration

Create `configs/segformer_config.yaml`:

```yaml
# Model Configuration
model:
  name: "nvidia/segformer-b0-finetuned-ade-512-512"  # Pretrained SegFormer B0
  num_classes: 4  # background + die + scratches + contamination
  
# Data Configuration
data:
  input_shape: 640  # Will be adjusted to multiple of 32 internally
  train_split: 0.8
  val_split: 0.2
  
# Augmentation Configuration
augmentation:
  # Static augmentations (every batch)
  static:
    rotate_angles: [0, 90, 180, 270]
    flip_x: true
    flip_y: true
    flip_xy: true
    random_crop: true
    crop_size: 224  # input_shape - 32
    
  # Curriculum learning augmentations (every N epochs)
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
    border_mode: "mirror"  # For geometric transforms
    
# Training Configuration
training:
  batch_size: 4  # Reduce for CPU
  num_epochs: 50
  learning_rate: 0.00005  # Lower LR for fine-tuning
  weight_decay: 0.01
  scheduler: "cosine"
  warmup_epochs: 5
  
# Evaluation
evaluation:
  metrics: ["iou", "dice", "pixel_accuracy", "class_accuracy"]
  save_predictions: true
  
# Hardware
hardware:
  device: "cpu"
  num_workers: 2  # Low for CPU
```

---

## 🔄 4. Augmentation Strategy

Our augmentation pipeline uses **Curriculum Learning**:

### Phase 1: Static Augmentations (Every Batch)
These are applied consistently throughout training:
- **Rotations**: 0°, 90°, 180°, 270°
- **Flips**: Horizontal, Vertical, Both
- **Random Crop**: 224x224 from 640x640 (minimal distortion)
- **Mirror Border**: Prevents edge artifacts

### Phase 2: Dynamic Augmentations (Every N Epochs)
Applied periodically to increase difficulty:
- **Small XY Shift**: ±10 pixels
- **Small Rotation**: ±15°
- **Gaussian Blur**: Kernel 3-5, σ 0.5-2.0
- **Sharpening**: Unsharp mask strength 0.2-0.5
- **Intensity Offset**: ±20 grayscale values
- **Contrast Stretch**: 0.8x to 1.2x scaling

### Additional AI-Recommended Augmentations
- **Elastic Deformations**: Simulate physical warping
- **Color Jitter**: Subtle changes for SWIR variations
- **Noise Injection**: Gaussian/Poisson noise
- **Cutout/MixUp**: Advanced regularization

---

## 🚀 5. Training Script

### Main Training Script: `train_segformer.py`

```python
#!/usr/bin/env python3
"""
SegFormer Training Script for SWIR Vitrox Board Inspection
Supports curriculum learning with progressive augmentations
"""

import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import torch.nn.functional as F

from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm
from sklearn.metrics import jaccard_score, f1_score
import matplotlib.pyplot as plt


class SWIRDataset(Dataset):
    """Custom dataset for SWIR board inspection"""
    
    def __init__(self, image_dir: str, mask_dir: str, 
                 input_shape: int = 640, 
                 augmentation_type: str = "static",
                 epoch: int = 0,
                 config: dict = None):
        
        self.image_paths = list(Path(image_dir).glob("*.png"))
        self.mask_paths = [Path(mask_dir) / p.name for p in self.image_paths]
        self.input_shape = input_shape
        self.augmentation_type = augmentation_type
        self.epoch = epoch
        self.config = config
        
        # Adjust to multiple of 32 for SegFormer
        self.processed_shape = (input_shape // 32) * 32
        
        self.transforms_static = self._get_static_transforms()
        self.transforms_curriculum = self._get_curriculum_transforms()
        
    def _get_static_transforms(self) -> A.Compose:
        """Static augmentations applied every batch"""
        return A.Compose([
            A.OneOf([
                A.Rotate(limit=[0], p=1),
                A.Rotate(limit=[90], p=1),
                A.Rotate(limit=[180], p=1),
                A.Rotate(limit=[270], p=1),
            ], p=0.7),
            
            A.OneOf([
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
                A.Transpose(p=1),
                A.Identity(),
            ], p=0.7),
            
            A.RandomCrop(
                height=self.config['augmentation']['static'].get('crop_size', self.processed_shape - 32),
                width=self.config['augmentation']['static'].get('crop_size', self.processed_shape - 32),
                p=0.5
            ),
            
            A.Resize(self.processed_shape, self.processed_shape),
            
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], is_check_shapes=False)
    
    def _get_curriculum_transforms(self) -> A.Compose:
        """Dynamic augmentations applied every N epochs"""
        cfg = self.config['augmentation']['curriculum']
        return A.Compose([
            A.ShiftScaleRotate(
                shift_limit=cfg.get('small_shift_range', 10) / self.processed_shape,
                rotate_limit=cfg.get('small_rotation_range', 15),
                scale_limit=0,
                border_mode=cv2.BORDER_REFLECT_101,  # Mirror border
                p=0.8
            ),
            
            A.OneOf([
                A.GaussianBlur(
                    blur_limit=tuple(cfg.get('gaussian_blur_kernel', [3, 5])),
                    sigma_limit=tuple(cfg.get('gaussian_blur_sigma', [0.5, 2.0])),
                    p=1
                ),
                A.MotionBlur(blur_limit=3, p=1),
            ], p=0.5),
            
            A.UnsharpMask(
                radius=3,
                alpha=tuple(cfg.get('unsharp_mask_strength', [0.2, 0.5])),
                threshold=10,
                p=0.5
            ),
            
            A.RandomBrightnessContrast(
                brightness_limit=tuple(cfg.get('intensity_offset', [-20, 20])) / 255.0,
                contrast_limit=tuple(cfg.get('contrast_scale', [0.8, 1.2])),
                p=0.7
            ),
            
            # Additional AI-recommended augmentations
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=0,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.3
            ),
            
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
            
        ], is_check_shapes=False)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load image and mask
        image = cv2.imread(str(self.image_paths[idx]), cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_UNCHANGED)
        
        # Apply augmentations
        if self.augmentation_type == "both" or self.augmentation_type == "static":
            transformed = self.transforms_static(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']
        
        # Apply curriculum augmentations if epoch matches interval
        if (self.augmentation_type == "both" or self.augmentation_type == "curriculum") and \
           self.config['augmentation']['curriculum']['enabled'] and \
           self.epoch % self.config['augmentation']['curriculum']['interval'] == 0:
            
            # Re-load original for curriculum transforms
            image_orig = cv2.imread(str(self.image_paths[idx]), cv2.IMREAD_GRAYSCALE)
            image_orig = cv2.cvtColor(image_orig, cv2.COLOR_GRAY2RGB)
            mask_orig = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_UNCHANGED)
            
            transformed_curr = self.transforms_curriculum(image=image_orig, mask=mask_orig)
            # Blend with static augmented version
            image = (image.permute(1, 2, 0).numpy() * 0.5 + 
                    transformed_curr['image'].permute(1, 2, 0).numpy() * 0.5)
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.tensor(mask_orig, dtype=torch.long)
        
        return {
            'pixel_values': image,
            'labels': mask,
            'image_path': str(self.image_paths[idx])
        }


class SegFormerTrainer:
    """Trainer class for SegFormer model"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(self.config['hardware']['device'])
        self.num_classes = self.config['model']['num_classes']
        
        # Load pretrained model
        print(f"Loading model: {self.config['model']['name']}")
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            self.config['model']['name'],
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True
        ).to(self.device)
        
        # Image processor
        self.processor = SegformerImageProcessor.from_pretrained(
            self.config['model']['name']
        )
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['num_epochs'] - self.config['training']['warmup_epochs'],
            eta_min=1e-6
        )
        
        # Metrics storage
        self.train_losses = []
        self.val_losses = []
        self.train_ious = []
        self.val_ious = []
        
    def compute_metrics(self, pred_masks: torch.Tensor, true_masks: torch.Tensor) -> Dict[str, float]:
        """Compute segmentation metrics"""
        pred_masks = pred_masks.argmax(dim=1).cpu().numpy()
        true_masks = true_masks.cpu().numpy()
        
        # Flatten for calculation
        pred_flat = pred_masks.flatten()
        true_flat = true_masks.flatten()
        
        # IoU per class
        iou_per_class = jaccard_score(true_flat, pred_flat, average=None, 
                                       labels=list(range(self.num_classes)),
                                       zero_division=0)
        
        # Dice score
        dice_scores = []
        for cls in range(self.num_classes):
            pred_cls = (pred_flat == cls)
            true_cls = (true_flat == cls)
            intersection = np.logical_and(pred_cls, true_cls).sum()
            union = pred_cls.sum() + true_cls.sum()
            dice = 2 * intersection / union if union > 0 else 0
            dice_scores.append(dice)
        
        # Pixel accuracy
        pixel_acc = (pred_flat == true_flat).mean()
        
        return {
            'mean_iou': np.mean(iou_per_class),
            'iou_per_class': iou_per_class.tolist(),
            'mean_dice': np.mean(dice_scores),
            'pixel_accuracy': pixel_acc
        }
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Tuple[float, Dict]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
        for batch in pbar:
            pixel_values = batch['pixel_values'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            all_preds.append(outputs.logits.detach().cpu())
            all_targets.append(labels.cpu())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self.compute_metrics(all_preds, all_targets)
        
        return total_loss / len(dataloader), metrics
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Tuple[float, Dict]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(dataloader, desc="Validating")
        for batch in pbar:
            pixel_values = batch['pixel_values'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item()
            all_preds.append(outputs.logits.cpu())
            all_targets.append(labels.cpu())
        
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self.compute_metrics(all_preds, all_targets)
        
        return total_loss / len(dataloader), metrics
    
    def train(self, train_dataset: Dataset, val_dataset: Dataset):
        """Full training loop with curriculum learning"""
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['hardware']['num_workers'],
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['hardware']['num_workers']
        )
        
        print(f"Starting training for {self.config['training']['num_epochs']} epochs...")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.config['training']['batch_size']}")
        
        best_val_iou = 0
        
        for epoch in range(self.config['training']['num_epochs']):
            # Update dataset epoch for curriculum learning
            train_dataset.epoch = epoch
            
            # Determine augmentation type based on epoch
            if (self.config['augmentation']['curriculum']['enabled'] and 
                epoch % self.config['augmentation']['curriculum']['interval'] == 0 and
                epoch > 0):
                train_dataset.augmentation_type = "both"
                print(f"\n📈 Epoch {epoch+1}: Applying curriculum augmentations!")
            else:
                train_dataset.augmentation_type = "static"
            
            # Training
            train_loss, train_metrics = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            self.train_ious.append(train_metrics['mean_iou'])
            
            # Validation
            val_loss, val_metrics = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_ious.append(val_metrics['mean_iou'])
            
            # Learning rate scheduling
            if epoch >= self.config['training']['warmup_epochs']:
                self.scheduler.step()
            
            # Print epoch summary
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.config['training']['num_epochs']}")
            print(f"Train Loss: {train_loss:.4f} | Train mIoU: {train_metrics['mean_iou']:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val mIoU: {val_metrics['mean_iou']:.4f}")
            print(f"IoU per class: {[f'{x:.3f}' for x in train_metrics['iou_per_class']]}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"{'='*60}\n")
            
            # Save best model
            if val_metrics['mean_iou'] > best_val_iou:
                best_val_iou = val_metrics['mean_iou']
                self.save_model(f"best_segformer_swir.pth")
                print(f"💾 Saved best model with mIoU: {best_val_iou:.4f}")
        
        # Plot training curves
        self.plot_training_curves()
        
        # Final evaluation
        print(f"\n🎉 Training completed! Best validation mIoU: {best_val_iou:.4f}")
        
    def save_model(self, path: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        axes[0].plot(self.train_losses, label='Train Loss', marker='o')
        axes[0].plot(self.val_losses, label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training & Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # IoU curves
        axes[1].plot(self.train_ious, label='Train mIoU', marker='o')
        axes[1].plot(self.val_ious, label='Val mIoU', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('mIoU')
        axes[1].set_title('Training & Validation mIoU')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves_segformer.png', dpi=300)
        plt.show()
        print("📊 Training curves saved to 'training_curves_segformer.png'")


def main():
    parser = argparse.ArgumentParser(description='Train SegFormer for SWIR inspection')
    parser.add_argument('--config', type=str, default='configs/segformer_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--train_images', type=str, required=True,
                       help='Path to training images directory')
    parser.add_argument('--train_masks', type=str, required=True,
                       help='Path to training masks directory')
    parser.add_argument('--val_images', type=str, required=True,
                       help='Path to validation images directory')
    parser.add_argument('--val_masks', type=str, required=True,
                       help='Path to validation masks directory')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create datasets
    train_dataset = SWIRDataset(
        image_dir=args.train_images,
        mask_dir=args.train_masks,
        input_shape=config['data']['input_shape'],
        config=config
    )
    
    val_dataset = SWIRDataset(
        image_dir=args.val_images,
        mask_dir=args.val_masks,
        input_shape=config['data']['input_shape'],
        augmentation_type="static",  # No curriculum for validation
        config=config
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize trainer
    trainer = SegFormerTrainer(args.config)
    
    # Start training
    trainer.train(train_dataset, val_dataset)


if __name__ == "__main__":
    main()
```

---

## 🏃 6. Running Training

### Step 1: Prepare Your Data
```bash
# Organize your data
mkdir -p data/processed/images/{train,val}
mkdir -p data/processed/masks/{train,val}

# Run conversion script (after labeling in CVAT)
python scripts/convert_coco_to_masks.py \
    --coco_json data/cvat_annotations/annotations.json \
    --images_dir data/raw_images \
    --output_dir data/processed \
    --split_ratio 0.8
```

### Step 2: Start Training
```bash
python train_segformer.py \
    --config configs/segformer_config.yaml \
    --train_images data/processed/images/train \
    --train_masks data/processed/masks/train \
    --val_images data/processed/images/val \
    --val_masks data/processed/masks/val
```

### Expected Output
```
Loading model: nvidia/segformer-b0-finetuned-ade-512-512
Training samples: 156
Validation samples: 39
Starting training for 50 epochs...
Device: cpu
Batch size: 4

============================================================
Epoch 1/50
Train Loss: 0.8234 | Train mIoU: 0.4521
Val Loss: 0.7891 | Val mIoU: 0.4832
IoU per class: ['0.921', '0.654', '0.312', '0.421']
Learning Rate: 0.000050
============================================================

📈 Epoch 2: Applying curriculum augmentations!
...
💾 Saved best model with mIoU: 0.7234
```

---

## 📊 7. Model Evaluation

### Inference Script: `evaluate_segformer.py`

```python
#!/usr/bin/env python3
"""
Evaluation script for trained SegFormer model
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import json


class SegFormerEvaluator:
    def __init__(self, model_path: str, config_path: str):
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.device = torch.device('cpu')
        self.num_classes = self.config['model']['num_classes']
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            self.config['model']['name'],
            num_labels=self.num_classes
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Processor
        self.processor = SegformerImageProcessor.from_pretrained(
            self.config['model']['name']
        )
        
        # Class names
        self.class_names = {
            0: 'background',
            1: 'die',
            2: 'scratches',
            3: 'contamination'
        }
    
    def predict(self, image_path: str) -> np.ndarray:
        """Run inference on single image"""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        inputs = self.processor(images=image_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        pred_mask = outputs.logits.argmax(dim=1).squeeze().cpu().numpy()
        return pred_mask
    
    def evaluate_dataset(self, image_dir: str, mask_dir: str):
        """Evaluate on entire dataset"""
        image_paths = list(Path(image_dir).glob("*.png"))
        mask_paths = [Path(mask_dir) / p.name for p in image_paths]
        
        all_preds = []
        all_trues = []
        
        print(f"Evaluating {len(image_paths)} images...")
        
        for img_path, mask_path in zip(image_paths, mask_paths):
            pred = self.predict(str(img_path))
            true = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            
            all_preds.extend(pred.flatten())
            all_trues.extend(true.flatten())
        
        all_preds = np.array(all_preds)
        all_trues = np.array(all_trues)
        
        # Classification report
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(
            all_trues, all_preds,
            target_names=[self.class_names[i] for i in range(self.num_classes)],
            digits=4
        ))
        
        # Confusion matrix
        cm = confusion_matrix(all_trues, all_preds, labels=list(range(self.num_classes)))
        
        # Per-class IoU
        ious = []
        for i in range(self.num_classes):
            intersection = np.logical_and(all_trues == i, all_preds == i).sum()
            union = np.logical_or(all_trues == i, all_preds == i).sum()
            iou = intersection / union if union > 0 else 0
            ious.append(iou)
            print(f"IoU for {self.class_names[i]}: {iou:.4f}")
        
        print(f"\nMean IoU: {np.mean(ious):.4f}")
        
        return cm, ious
    
    def visualize_prediction(self, image_path: str, output_path: str = None):
        """Visualize prediction vs ground truth"""
        image = cv2.imread(image_path)
        mask_path = str(Path(image_path).parent.parent / "masks" / Path(image_path).name)
        true_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        pred_mask = self.predict(image_path)
        
        # Create color maps
        colors = [
            [0, 0, 0],      # background - black
            [255, 0, 0],    # die - red
            [0, 255, 0],    # scratches - green
            [0, 0, 255],    # contamination - blue
        ]
        
        pred_color = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        true_color = np.zeros((*true_mask.shape, 3), dtype=np.uint8)
        
        for i in range(self.num_classes):
            pred_color[pred_mask == i] = colors[i]
            true_color[true_mask == i] = colors[i]
        
        # Blend with original image
        pred_blend = cv2.addWeighted(image, 0.6, pred_color, 0.4, 0)
        true_blend = cv2.addWeighted(image, 0.6, true_color, 0.4, 0)
        
        # Display
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(true_blend)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        axes[2].imshow(pred_blend)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Evaluate SegFormer model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file (saved with model)')
    parser.add_argument('--images', type=str, required=True,
                       help='Path to test images directory')
    parser.add_argument('--masks', type=str, required=True,
                       help='Path to test masks directory')
    parser.add_argument('--visualize', type=str, default=None,
                       help='Path to single image for visualization')
    
    args = parser.parse_args()
    
    evaluator = SegFormerEvaluator(args.model, args.config)
    
    if args.visualize:
        evaluator.visualize_prediction(args.visualize, 'prediction_visualization.png')
    else:
        evaluator.evaluate_dataset(args.images, args.masks)


if __name__ == "__main__":
    main()
```

### Run Evaluation
```bash
# Full dataset evaluation
python evaluate_segformer.py \
    --model best_segformer_swir.pth \
    --config config_backup.json \
    --images data/processed/images/val \
    --masks data/processed/masks/val

# Visualize single prediction
python evaluate_segformer.py \
    --model best_segformer_swir.pth \
    --config config_backup.json \
    --visualize data/processed/images/val/sample_001.png
```

---

## 🛠️ 8. Helper Scripts

### COCO to Mask Converter: `scripts/convert_coco_to_masks.py`

```python
#!/usr/bin/env python3
"""
Convert CVAT COCO JSON annotations to indexed PNG masks
"""

import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import cv2


def coco_to_masks(coco_json: str, images_dir: str, output_dir: str, 
                  class_mapping: dict, split_ratio: float = 0.8):
    """Convert COCO annotations to indexed masks"""
    
    # Load COCO annotations
    with open(coco_json, 'r') as f:
        coco_data = json.load(f)
    
    # Create reverse class mapping (name -> id)
    name_to_id = {v: int(k) for k, v in class_mapping.items()}
    
    # Get all images
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
    np.random.shuffle(image_ids)
    split_idx = int(len(image_ids) * split_ratio)
    train_ids = image_ids[:split_idx]
    val_ids = image_ids[split_idx:]
    
    # Create output directories
    for split in ['train', 'val']:
        Path(f"{output_dir}/images/{split}").mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/masks/{split}").mkdir(parents=True, exist_ok=True)
    
    def process_image(img_id: int, split: str):
        img_info = images[img_id]
        img_name = img_info['file_name']
        
        # Load image
        img_path = Path(images_dir) / img_name
        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            return
        
        image = Image.open(img_path)
        width, height = image.size
        
        # Create blank mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Draw annotations
        if img_id in img_annotations:
            for ann in img_annotations[img_id]:
                category_id = ann['category_id']
                
                # Find class name from COCO categories
                cat_name = None
                for cat in coco_data['categories']:
                    if cat['id'] == category_id:
                        cat_name = cat['name']
                        break
                
                if cat_name and cat_name in name_to_id:
                    class_id = name_to_id[cat_name]
                    
                    # Get segmentation polygon
                    segmentation = ann['segmentation']
                    if isinstance(segmentation, list):
                        for poly in segmentation:
                            # Convert flat list to [(x,y), ...]
                            poly_points = [(poly[i], poly[i+1]) 
                                          for i in range(0, len(poly), 2)]
                            
                            # Draw polygon
                            draw = ImageDraw.Draw(mask)
                            draw.polygon(poly_points, fill=class_id)
        
        # Save image and mask
        image.save(f"{output_dir}/images/{split}/{img_name}")
        Image.fromarray(mask).save(f"{output_dir}/masks/{split}/{img_name}")
        
        print(f"Processed: {img_name} ({split})")
    
    # Process train and val sets
    for img_id in train_ids:
        process_image(img_id, 'train')
    
    for img_id in val_ids:
        process_image(img_id, 'val')
    
    print(f"\nConversion complete!")
    print(f"Train: {len(train_ids)} images")
    print(f"Val: {len(val_ids)} images")


def main():
    parser = argparse.ArgumentParser(description='Convert COCO to indexed masks')
    parser.add_argument('--coco_json', type=str, required=True,
                       help='Path to COCO JSON file from CVAT')
    parser.add_argument('--images_dir', type=str, required=True,
                       help='Directory containing original images')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for processed data')
    parser.add_argument('--class_mapping', type=str, required=True,
                       help='JSON string: {"0": "background", "1": "die", ...}')
    parser.add_argument('--split_ratio', type=float, default=0.8,
                       help='Train/val split ratio (default: 0.8)')
    
    args = parser.parse_args()
    
    class_mapping = json.loads(args.class_mapping)
    
    coco_to_masks(
        args.coco_json,
        args.images_dir,
        args.output_dir,
        class_mapping,
        args.split_ratio
    )


if __name__ == "__main__":
    main()
```

---

## 📈 9. Tips for Better Results

### Data Quality
- Ensure annotations are precise and consistent
- Include diverse defect examples in training set
- Balance classes if possible (use weighted loss if imbalanced)

### Hyperparameter Tuning
- **Learning Rate**: Try 1e-4 to 5e-5 for fine-tuning
- **Batch Size**: Increase if memory allows (even on CPU with smaller images)
- **Input Shape**: 640x640 is good balance; try 512x512 for faster iteration

### Augmentation Strategy
- Start with static augmentations only
- Enable curriculum learning after baseline is established
- Monitor validation loss to avoid over-augmentation

### Debugging
- Visualize augmented batches before training
- Check class distribution in masks
- Verify image-mask alignment

---

## 🔧 10. Troubleshooting

### Common Issues

**Issue**: Out of memory (even on CPU)
- **Solution**: Reduce batch_size to 1 or 2, decrease input_shape

**Issue**: Model not converging
- **Solution**: Lower learning rate, check data quality, verify class balance

**Issue**: Poor performance on small defects
- **Solution**: Increase resolution, add more augmentation for small objects, use focal loss

**Issue**: Slow training on CPU
- **Solution**: Use smaller model (SegFormer-B0), reduce image size, enable mixed precision if GPU becomes available

---

## 📚 11. Next Steps

1. **Label your data** using CVAT
2. **Run conversion script** to prepare datasets
3. **Start with baseline training** (static augmentations only)
4. **Enable curriculum learning** once baseline is stable
5. **Evaluate and iterate** on model performance
6. **Deploy model** for inference on new boards

---

## 📞 Support

For questions about:
- **CVAT labeling**: Request separate CVAT tutorial
- **Data format issues**: Check conversion script output
- **Training problems**: Review logs and visualization
- **Model deployment**: Contact for inference optimization guide

Good luck with your SWIR Vitrox board inspection project! 🚀
