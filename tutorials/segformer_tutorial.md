# SegFormer Training Tutorial for SWIR Vitrox Board Inspection

## 🎯 Overview
This tutorial guides you through training a **SegFormer** segmentation model for inspecting SWIR Vitrox boards. We'll cover data preparation, augmentation strategies with curriculum learning, training, and evaluation.

**Why SegFormer?**
- Transformer-based architecture with excellent accuracy
- Handles complex textures and small defects well
- Great for industrial inspection tasks
- Supports various input sizes (multiples of 32)

---

## 📋 Prerequisites

### System Requirements
- **Hardware**: CPU (as specified), 16GB+ RAM recommended
- **Python**: 3.8+
- **Package Manager**: conda

### Installation Steps

```bash
# Create conda environment
conda create -n segformer_inspection python=3.9 -y
conda activate segformer_inspection

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers albumentations opencv-python pillow numpy
pip install scikit-image matplotlib tqdm

# Install additional utilities
pip install pandas scipy
```

### Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from transformers import SegformerForSemanticSegmentation; print('Transformers OK')"
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
├── masks_indexed/           # Indexed mask PNGs from CVAT
│   ├── img_001_mask.png     # Pixel values: 0=background, 1=die, 2=scratches, etc.
│   ├── img_002_mask.png
│   └── ...
├── splits/
│   ├── train.txt            # List of training image filenames
│   ├── val.txt              # List of validation image filenames
│   └── test.txt             # List of test image filenames
└── processed/               # Will be created by preprocessing script
    ├── train/
    │   ├── images/
    │   └── masks/
    └── val/
        ├── images/
        └── masks/
```

### Class Mapping
Create a `class_mapping.json` file:
```json
{
  "0": "background",
  "1": "die",
  "2": "scratches",
  "3": "contamination",
  "4": "other_defects"
}
```

### Exporting from CVAT
1. In CVAT, create your dataset with the following labels:
   - background
   - die
   - scratches
   - contamination
   - other_defects (optional)

2. Export format: **CVAT for images 1.1** or **Mask RGB**
3. Convert to indexed masks using the provided conversion script

---

## 🔄 Data Conversion Script

Save as `scripts/convert_cvat_to_indexed.py`:

```python
#!/usr/bin/env python3
"""
Convert CVAT export to indexed mask format for SegFormer training.
"""

import os
import json
import argparse
import numpy as np
from PIL import Image
import cv2

def create_class_mapping():
    """Define class to index mapping."""
    return {
        'background': 0,
        'die': 1,
        'scratches': 2,
        'contamination': 3,
        'other_defects': 4
    }

def convert_cvat_masks(cvat_dir, output_dir, class_mapping):
    """
    Convert CVAT mask exports to indexed single-channel masks.
    
    Args:
        cvat_dir: Directory containing CVAT exported masks
        output_dir: Output directory for indexed masks
        class_mapping: Dictionary mapping class names to indices
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Reverse mapping for color to index
    # CVAT typically uses RGB colors for each class
    color_map = {
        (0, 0, 0): class_mapping['background'],
        (128, 0, 0): class_mapping['die'],  # Example colors - adjust based on your CVAT setup
        (0, 128, 0): class_mapping['scratches'],
        (0, 0, 128): class_mapping['contamination'],
        (128, 128, 0): class_mapping['other_defects']
    }
    
    mask_files = [f for f in os.listdir(cvat_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for mask_file in mask_files:
        mask_path = os.path.join(cvat_dir, mask_file)
        mask_rgb = cv2.imread(mask_path)
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
        
        # Create indexed mask
        indexed_mask = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
        
        for color, class_idx in color_map.items():
            mask_channel = np.all(mask_rgb == color, axis=-1)
            indexed_mask[mask_channel] = class_idx
        
        # Save indexed mask
        output_path = os.path.join(output_dir, mask_file)
        Image.fromarray(indexed_mask).save(output_path)
        print(f"Converted: {mask_file}")
    
    print(f"Total converted: {len(mask_files)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CVAT masks to indexed format")
    parser.add_argument("--cvat_dir", type=str, required=True, help="Input directory with CVAT masks")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for indexed masks")
    args = parser.parse_args()
    
    class_mapping = create_class_mapping()
    convert_cvat_masks(args.cvat_dir, args.output_dir, class_mapping)
```

### Usage:
```bash
python scripts/convert_cvat_to_indexed.py \
    --cvat_dir data/cvat_masks \
    --output_dir data/masks_indexed
```

---

## 🎨 Augmentation Strategy with Curriculum Learning

### Configuration File

Save as `configs/segformer_config.yaml`:

```yaml
# SegFormer Training Configuration for SWIR Board Inspection

# Model Settings
model:
  checkpoint: "nvidia/segformer-b0-finetuned-ade-512-512"  # or b1, b2, b3, b4, b5
  num_labels: 5  # background, die, scratches, contamination, other_defects
  ignore_index: 255

# Data Settings
data:
  train_images_dir: "data/processed/train/images"
  train_masks_dir: "data/processed/train/masks"
  val_images_dir: "data/processed/val/images"
  val_masks_dir: "data/processed/val/masks"
  input_size: 640  # Must be multiple of 32 for SegFormer
  num_workers: 4

# Augmentation Settings
augmentation:
  # Static augmentations (applied every epoch)
  static:
    rotate_angles: [0, 90, 180, 270]
    flip_options: ["none", "horizontal", "vertical", "both"]
    random_crop_enabled: true
    crop_size_ratio: 0.9  # Crop to 90% of original size
  
  # Dynamic augmentations (curriculum learning - applied every N epochs)
  dynamic:
    apply_every_n_epochs: 2
    enabled: true
    
    # Small geometric transformations
    xy_shift_range: [-10, 10]  # pixels
    rotation_range: [-5, 5]  # degrees
    
    # Image quality augmentations
    gaussian_blur_sigma: [0.5, 1.5]
    sharpen_alpha: [0.5, 1.5]
    unsharp_kernel_size: 3
    
    # Intensity augmentations
    intensity_offset_range: [-15, 15]
    contrast_scale_range: [0.85, 1.15]
    gamma_range: [0.9, 1.1]
    
    # Additional recommended augmentations
    elastic_transform_enabled: true
    elastic_alpha: 50
    elastic_sigma: 5
    grid_distortion_enabled: true
    grid_num_steps: 5
    grid_distort_limit: 0.05
    optical_distortion_enabled: true
    optical_distort_limit: 0.05
    
    # Noise augmentations
    gauss_noise_var_limit: [5.0, 20.0]
    speckle_noise_enabled: true
    speckle_var_range: [0.05, 0.2]
    
    # Weather/degradation simulations (optional)
    motion_blur_enabled: true
    motion_blur_limit: 3
    defocus_blur_enabled: true
    defocus_radius: [1, 3]
  
  # Border handling
  border_mode: "reflect"  # mirror border for XY augmentations
  
  # Probability settings
  prob_static_aug: 0.8  # Probability of applying static augmentations
  prob_dynamic_aug: 0.6  # Probability of applying dynamic augmentations when active

# Training Settings
training:
  batch_size: 4  # Reduce for CPU training
  num_epochs: 50
  learning_rate: 0.00005  # Lower LR for fine-tuning transformers
  weight_decay: 0.01
  scheduler: "cosine"  # cosine, linear, or step
  warmup_epochs: 5
  early_stopping_patience: 10
  gradient_accumulation_steps: 2  # Simulate larger batch on CPU
  
  # Checkpointing
  save_every_n_epochs: 5
  checkpoint_dir: "checkpoints/segformer"
  
  # Mixed precision (not available on CPU, but kept for future GPU use)
  use_amp: false

# Evaluation Settings
evaluation:
  metrics: ["miou", "pixel_accuracy", "dice_coefficient", "precision", "recall"]
  save_predictions: true
  visualization_samples: 10

# Random Seed for Reproducibility
seed: 42
```

---

## 🚀 Training Script

Save as `scripts/train_segformer.py`:

```python
#!/usr/bin/env python3
"""
SegFormer Training Script with Curriculum Learning for SWIR Board Inspection.
Supports CPU training with configurable augmentations.
"""

import os
import yaml
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from datetime import datetime

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SWIRBoardDataset(Dataset):
    """Custom dataset for SWIR board inspection with curriculum learning augmentations."""
    
    def __init__(self, images_dir, masks_dir, config, epoch=0, is_train=True):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.config = config
        self.epoch = epoch
        self.is_train = is_train
        
        self.image_files = sorted([f for f in os.listdir(images_dir) 
                                   if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.mask_files = sorted([f for f in os.listdir(masks_dir) 
                                  if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        assert len(self.image_files) == len(self.mask_files), \
            "Number of images and masks must match"
        
        self.input_size = config['data']['input_size']
        self.num_labels = config['model']['num_labels']
        
        # Build augmentations
        self.static_transform = self._build_static_transform()
        self.dynamic_transform = self._build_dynamic_transform()
        
        # Image processor from Hugging Face
        self.image_processor = SegformerImageProcessor.from_pretrained(
            config['model']['checkpoint'],
            size={"height": self.input_size, "width": self.input_size},
            do_rescale=False
        )
    
    def _build_static_transform(self):
        """Build static augmentations applied every epoch."""
        aug_config = self.config['augmentation']['static']
        
        transforms = []
        
        # Random crop (minimal distortion)
        if aug_config.get('random_crop_enabled', True):
            crop_size = int(self.input_size * aug_config.get('crop_size_ratio', 0.9))
            transforms.append(A.RandomCrop(height=crop_size, width=crop_size, p=0.5))
        
        # Rotate 0, 90, 180, 270
        transforms.append(A.Rotate(limit=0, p=0.25))  # Placeholder, custom rotation below
        
        # Flip X, Y, XY
        transforms.append(A.HorizontalFlip(p=0.25))
        transforms.append(A.VerticalFlip(p=0.25))
        
        # Resize to input size
        transforms.append(A.Resize(height=self.input_size, width=self.input_size))
        
        # Normalize for transformer
        transforms.append(A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ))
        
        transforms.append(ToTensorV2())
        
        return A.Compose(transforms, bbox_params=None, keypoint_params=None)
    
    def _build_dynamic_transform(self):
        """Build dynamic augmentations for curriculum learning."""
        aug_config = self.config['augmentation']['dynamic']
        
        transforms = []
        
        # Small XY shift
        xy_shift = aug_config.get('xy_shift_range', [-10, 10])
        transforms.append(A.Affine(
            translate_percent={"x": (xy_shift[0]/self.input_size, xy_shift[1]/self.input_size),
                              "y": (xy_shift[0]/self.input_size, xy_shift[1]/self.input_size)},
            p=0.5
        ))
        
        # Small rotation
        rot_range = aug_config.get('rotation_range', [-5, 5])
        transforms.append(A.Affine(rotate=rot_range, p=0.5))
        
        # Gaussian blur
        blur_sigma = aug_config.get('gaussian_blur_sigma', [0.5, 1.5])
        transforms.append(A.GaussianBlur(blur_limit=(3, 7), sigma_limit=blur_sigma, p=0.3))
        
        # Sharpening (Unsharp Mask)
        sharp_alpha = aug_config.get('sharpen_alpha', [0.5, 1.5])
        transforms.append(A.Sharpen(alpha=sharp_alpha, lightness=(0.8, 1.2), p=0.3))
        
        # Intensity offsets
        intensity_range = aug_config.get('intensity_offset_range', [-15, 15])
        transforms.append(A.Lambda(
            name="intensity_offset",
            apply=self._apply_intensity_offset,
            p=0.4
        ))
        
        # Contrast stretch (scaling)
        contrast_range = aug_config.get('contrast_scale_range', [0.85, 1.15])
        transforms.append(A.Lambda(
            name="contrast_stretch",
            apply=self._apply_contrast_stretch,
            p=0.4
        ))
        
        # Gamma correction
        gamma_range = aug_config.get('gamma_range', [0.9, 1.1])
        transforms.append(A.RandomGamma(gamma_limit=gamma_range, p=0.3))
        
        # Elastic transform (recommended for defect detection)
        if aug_config.get('elastic_transform_enabled', True):
            transforms.append(A.ElasticTransform(
                alpha=aug_config.get('elastic_alpha', 50),
                sigma=aug_config.get('elastic_sigma', 5),
                p=0.2
            ))
        
        # Grid distortion
        if aug_config.get('grid_distortion_enabled', True):
            transforms.append(A.GridDistortion(
                num_steps=aug_config.get('grid_num_steps', 5),
                distort_limit=aug_config.get('grid_distort_limit', 0.05),
                p=0.2
            ))
        
        # Optical distortion
        if aug_config.get('optical_distortion_enabled', True):
            transforms.append(A.OpticalDistortion(
                distort_limit=aug_config.get('optical_distort_limit', 0.05),
                p=0.2
            ))
        
        # Gaussian noise
        noise_var = aug_config.get('gauss_noise_var_limit', [5.0, 20.0])
        transforms.append(A.GaussNoise(var_limit=noise_var, p=0.3))
        
        # Speckle noise
        if aug_config.get('speckle_noise_enabled', True):
            transforms.append(A.Lambda(
                name="speckle_noise",
                apply=self._apply_speckle_noise,
                p=0.2
            ))
        
        # Motion blur
        if aug_config.get('motion_blur_enabled', True):
            transforms.append(A.MotionBlur(
                blur_limit=aug_config.get('motion_blur_limit', 3),
                p=0.2
            ))
        
        # Defocus blur
        if aug_config.get('defocus_blur_enabled', True):
            transforms.append(A.Defocus(
                radius=aug_config.get('defocus_radius', [1, 3]),
                p=0.2
            ))
        
        # Resize to input size
        transforms.append(A.Resize(height=self.input_size, width=self.input_size))
        
        # Normalize
        transforms.append(A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ))
        
        transforms.append(ToTensorV2())
        
        return A.Compose(transforms, bbox_params=None, keypoint_params=None)
    
    def _apply_intensity_offset(self, image, **params):
        """Apply random intensity offset."""
        offset_range = self.config['augmentation']['dynamic']['intensity_offset_range']
        offset = np.random.uniform(offset_range[0], offset_range[1])
        image = np.clip(image.astype(np.float32) + offset, 0, 255).astype(np.uint8)
        return image
    
    def _apply_contrast_stretch(self, image, **params):
        """Apply random contrast scaling."""
        scale_range = self.config['augmentation']['dynamic']['contrast_scale_range']
        scale = np.random.uniform(scale_range[0], scale_range[1])
        mean_val = np.mean(image)
        image = np.clip((image - mean_val) * scale + mean_val, 0, 255).astype(np.uint8)
        return image
    
    def _apply_speckle_noise(self, image, **params):
        """Apply speckle noise."""
        var_range = self.config['augmentation']['dynamic'].get('speckle_var_range', [0.05, 0.2])
        var = np.random.uniform(var_range[0], var_range[1])
        noise = np.random.randn(*image.shape) * var * 255
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return image
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image and mask
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = np.array(Image.open(mask_path))
        
        # Apply augmentations based on epoch (curriculum learning)
        dynamic_period = self.config['augmentation']['dynamic']['apply_every_n_epochs']
        use_dynamic = (self.is_train and 
                      self.config['augmentation']['dynamic']['enabled'] and
                      self.epoch % dynamic_period == 0)
        
        if use_dynamic and random.random() < self.config['augmentation']['prob_dynamic_aug']:
            transformed = self.dynamic_transform(image=image, mask=mask)
        elif self.is_train and random.random() < self.config['augmentation']['prob_static_aug']:
            transformed = self.static_transform(image=image, mask=mask)
        else:
            # Minimal processing for validation or when no augmentation
            transform = A.Compose([
                A.Resize(height=self.input_size, width=self.input_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
                ToTensorV2()
            ])
            transformed = transform(image=image, mask=mask)
        
        pixel_values = transformed['image']
        labels = torch.tensor(mask, dtype=torch.long)
        
        return {
            'pixel_values': pixel_values,
            'labels': labels,
            'image_id': self.image_files[idx]
        }


def compute_metrics(predictions, labels, num_classes):
    """Compute segmentation metrics."""
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    # Ignore index (usually 255)
    valid_mask = labels != 255
    predictions = predictions[valid_mask]
    labels = labels[valid_mask]
    
    if len(labels) == 0:
        return {'miou': 0.0, 'pixel_accuracy': 0.0, 'dice': 0.0}
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions, labels=list(range(num_classes)))
    
    # Pixel Accuracy
    pixel_accuracy = np.diag(cm).sum() / cm.sum()
    
    # IoU per class
    ious = []
    for i in range(num_classes):
        intersection = cm[i, i]
        union = cm[i, :].sum() + cm[:, i].sum() - cm[i, i]
        if union > 0:
            ious.append(intersection / union)
        else:
            ious.append(0.0)
    
    miou = np.mean(ious)
    
    # Dice coefficient (average)
    dices = []
    for i in range(num_classes):
        intersection = cm[i, i]
        total = cm[i, :].sum() + cm[:, i].sum()
        if total > 0:
            dices.append(2 * intersection / total)
        else:
            dices.append(0.0)
    
    dice = np.mean(dices)
    
    return {
        'miou': miou,
        'pixel_accuracy': pixel_accuracy,
        'dice': dice,
        'iou_per_class': ious
    }


def train(config_path, args):
    """Main training function."""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.input_size:
        config['data']['input_size'] = args.input_size
    
    # Set seed
    set_seed(config.get('seed', 42))
    
    # Create checkpoint directory
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    
    # Initialize model
    print("Loading SegFormer model...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        config['model']['checkpoint'],
        num_labels=config['model']['num_labels'],
        ignore_index=config['model']['ignore_index']
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    # Create datasets
    train_dataset = SWIRBoardDataset(
        images_dir=config['data']['train_images_dir'],
        masks_dir=config['data']['train_masks_dir'],
        config=config,
        epoch=0,
        is_train=True
    )
    
    val_dataset = SWIRBoardDataset(
        images_dir=config['data']['val_images_dir'],
        masks_dir=config['data']['val_masks_dir'],
        config=config,
        epoch=0,
        is_train=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    if config['training']['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs']
        )
    elif config['training']['scheduler'] == 'linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=config['training']['num_epochs']
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.1
        )
    
    # Training loop
    best_miou = 0.0
    patience_counter = 0
    
    print(f"Starting training for {config['training']['num_epochs']} epochs...")
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        print(f"{'='*50}")
        
        # Update dataset epoch for curriculum learning
        train_dataset.epoch = epoch
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")
        for batch_idx, batch in enumerate(progress_bar):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            # Backward pass with gradient accumulation
            loss = loss / config['training']['gradient_accumulation_steps']
            loss.backward()
            
            if (batch_idx + 1) % config['training']['gradient_accumulation_steps'] == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * config['training']['gradient_accumulation_steps']
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val Epoch {epoch+1}"):
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()
                
                # Get predictions
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Compute metrics
        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)
        
        metrics = compute_metrics(
            all_predictions,
            all_labels,
            config['model']['num_labels']
        )
        
        print(f"\nTraining Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Mean IoU: {metrics['miou']:.4f}")
        print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
        print(f"Dice Coefficient: {metrics['dice']:.4f}")
        print(f"IoU per class: {[f'{x:.4f}' for x in metrics['iou_per_class']]}")
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_every_n_epochs'] == 0:
            checkpoint_path = os.path.join(
                config['training']['checkpoint_dir'],
                f"checkpoint_epoch_{epoch+1}.pth"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'miou': metrics['miou'],
                'config': config
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
        
        # Early stopping check
        if metrics['miou'] > best_miou:
            best_miou = metrics['miou']
            patience_counter = 0
            
            # Save best model
            best_model_path = os.path.join(
                config['training']['checkpoint_dir'],
                "best_model.pth"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'miou': best_miou,
                'config': config
            }, best_model_path)
            print(f"✨ New best model saved! mIoU: {best_miou:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config['training']['early_stopping_patience']:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    print("\n🎉 Training completed!")
    print(f"Best mIoU achieved: {best_miou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SegFormer for SWIR Board Inspection")
    parser.add_argument("--config", type=str, default="configs/segformer_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="Override learning rate")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Override batch size")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Override number of epochs")
    parser.add_argument("--input_size", type=int, default=None,
                       help="Override input size (must be multiple of 32)")
    
    args = parser.parse_args()
    train(args.config, args)
```

---

## 📊 Evaluation and Visualization

Save as `scripts/evaluate_segformer.py`:

```python
#!/usr/bin/env python3
"""
Evaluation script for SegFormer model with visualization.
"""

import os
import yaml
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import argparse

# Import dataset class from training script
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_segformer import SWIRBoardDataset, compute_metrics, set_seed

def visualize_predictions(model, dataloader, config, num_samples=10, device='cpu'):
    """Visualize model predictions vs ground truth."""
    
    model.eval()
    class_names = ['background', 'die', 'scratches', 'contamination', 'other_defects']
    colors = [
        [0, 0, 0],       # background - black
        [128, 0, 0],     # die - maroon
        [0, 128, 0],     # scratches - green
        [0, 0, 128],     # contamination - navy
        [128, 128, 0]    # other_defects - olive
    ]
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    samples_processed = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if samples_processed >= num_samples:
                break
            
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].cpu().numpy()
            image_ids = batch['image_id']
            
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            
            for i in range(len(pixel_values)):
                if samples_processed >= num_samples:
                    break
                
                # Original image (denormalize)
                img = pixel_values[i].cpu().numpy().transpose(1, 2, 0)
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
                
                # Ground truth mask
                gt_mask = labels[i]
                gt_colored = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
                for class_idx, color in enumerate(colors):
                    gt_colored[gt_mask == class_idx] = color
                
                # Prediction mask
                pred_mask = predictions[i]
                pred_colored = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
                for class_idx, color in enumerate(colors):
                    pred_colored[pred_mask == class_idx] = color
                
                # Difference map
                diff = np.zeros_like(gt_mask, dtype=np.float32)
                diff[gt_mask != pred_mask] = 1.0
                diff_colored = cv2.applyColorMap((diff * 255).astype(np.uint8), cv2.COLORMAP_JET)
                
                # Plot
                axes[samples_processed, 0].imshow(img)
                axes[samples_processed, 0].set_title(f"Image: {image_ids[i]}")
                axes[samples_processed, 0].axis('off')
                
                axes[samples_processed, 1].imshow(gt_colored)
                axes[samples_processed, 1].set_title("Ground Truth")
                axes[samples_processed, 1].axis('off')
                
                axes[samples_processed, 2].imshow(pred_colored)
                axes[samples_processed, 2].set_title("Prediction")
                axes[samples_processed, 2].axis('off')
                
                axes[samples_processed, 3].imshow(diff_colored)
                axes[samples_processed, 3].set_title("Difference")
                axes[samples_processed, 3].axis('off')
                
                samples_processed += 1
    
    plt.tight_layout()
    plt.savefig('evaluation_results/visualizations.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Visualizations saved to evaluation_results/visualizations.png")


def plot_confusion_matrix(all_predictions, all_labels, config):
    """Plot and save confusion matrix."""
    
    class_names = ['background', 'die', 'scratches', 'contamination', 'other_defects']
    cm = confusion_matrix(all_labels, all_predictions, labels=list(range(config['model']['num_labels'])))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('evaluation_results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Confusion matrix saved to evaluation_results/confusion_matrix.png")


def evaluate(config_path, checkpoint_path, args):
    """Main evaluation function."""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_seed(config.get('seed', 42))
    
    # Create output directory
    os.makedirs('evaluation_results', exist_ok=True)
    
    # Load model
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SegformerForSemanticSegmentation.from_pretrained(
        config['model']['checkpoint'],
        num_labels=config['model']['num_labels']
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Best mIoU: {checkpoint.get('miou', 'unknown')}")
    
    # Create test dataset
    test_dataset = SWIRBoardDataset(
        images_dir=config['data']['val_images_dir'],
        masks_dir=config['data']['val_masks_dir'],
        config=config,
        epoch=0,
        is_train=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    # Collect all predictions and labels
    all_predictions = []
    all_labels = []
    
    print("Running evaluation...")
    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels']
            
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_predictions.extend(predictions.flatten())
            all_labels.extend(labels.numpy().flatten())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Filter out ignore_index
    valid_mask = all_labels != 255
    all_predictions = all_predictions[valid_mask]
    all_labels = all_labels[valid_mask]
    
    # Compute metrics
    metrics = compute_metrics(
        all_predictions,
        all_labels,
        config['model']['num_labels']
    )
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Mean IoU: {metrics['miou']:.4f}")
    print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    print(f"Dice Coefficient: {metrics['dice']:.4f}")
    print("\nIoU per class:")
    class_names = ['background', 'die', 'scratches', 'contamination', 'other_defects']
    for i, iou in enumerate(metrics['iou_per_class']):
        print(f"  {class_names[i]:15s}: {iou:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, 
                                target_names=class_names,
                                digits=4))
    
    # Visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        visualize_predictions(model, test_loader, config, 
                            num_samples=args.num_samples, device=device)
        plot_confusion_matrix(all_predictions, all_labels, config)
    
    # Save metrics to file
    metrics_path = 'evaluation_results/metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write("SegFormer Evaluation Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"Mean IoU: {metrics['miou']:.4f}\n")
        f.write(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}\n")
        f.write(f"Dice Coefficient: {metrics['dice']:.4f}\n\n")
        f.write("IoU per class:\n")
        for i, iou in enumerate(metrics['iou_per_class']):
            f.write(f"  {class_names[i]:15s}: {iou:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(all_labels, all_predictions, 
                                     target_names=class_names,
                                     digits=4))
    
    print(f"\n✅ Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SegFormer model")
    parser.add_argument("--config", type=str, default="configs/segformer_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, required=True,
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
mkdir -p data/raw_images data/masks_indexed data/splits data/processed

# If you have CVAT exports, convert them
python scripts/convert_cvat_to_indexed.py \
    --cvat_dir data/cvat_masks \
    --output_dir data/masks_indexed

# Create train/val splits (80/20 split example)
cd data
ls raw_images | grep -E '\.(png|jpg|jpeg)$' > all_images.txt
shuf all_images.txt > shuffled.txt
head -n $(echo "$(wc -l < shuffled.txt) * 80 / 100" | bc) shuffled.txt > splits/train.txt
tail -n +$(echo "$(wc -l < shuffled.txt) * 80 / 100 + 1" | bc) shuffled.txt > splits/val.txt
```

### Step 2: Preprocess Data (Optional)

Create a simple script to organize data into train/val folders:

```bash
#!/bin/bash
# scripts/preprocess_data.sh

TRAIN_LIST="data/splits/train.txt"
VAL_LIST="data/splits/val.txt"

# Create directories
mkdir -p data/processed/train/images data/processed/train/masks
mkdir -p data/processed/val/images data/processed/val/masks

# Copy training data
while read filename; do
    cp "data/raw_images/$filename" "data/processed/train/images/"
    cp "data/masks_indexed/${filename%.*}_mask.png" "data/processed/train/masks/" 2>/dev/null || \
    cp "data/masks_indexed/$filename" "data/processed/train/masks/" 2>/dev/null
done < "$TRAIN_LIST"

# Copy validation data
while read filename; do
    cp "data/raw_images/$filename" "data/processed/val/images/"
    cp "data/masks_indexed/${filename%.*}_mask.png" "data/processed/val/masks/" 2>/dev/null || \
    cp "data/masks_indexed/$filename" "data/processed/val/masks/" 2>/dev/null
done < "$VAL_LIST"

echo "Data preprocessing complete!"
```

Make it executable and run:
```bash
chmod +x scripts/preprocess_data.sh
./scripts/preprocess_data.sh
```

### Step 3: Start Training

```bash
# Activate environment
conda activate segformer_inspection

# Train with default config
python scripts/train_segformer.py --config configs/segformer_config.yaml

# Or override parameters
python scripts/train_segformer.py \
    --config configs/segformer_config.yaml \
    --learning_rate 0.0001 \
    --batch_size 2 \
    --epochs 30 \
    --input_size 512
```

### Step 4: Evaluate the Model

```bash
# Evaluate best model
python scripts/evaluate_segformer.py \
    --config configs/segformer_config.yaml \
    --checkpoint checkpoints/segformer/best_model.pth \
    --visualize \
    --num_samples 15
```

---

## 📈 Monitoring Training

### Real-time Monitoring with TensorBoard (Optional)

Install TensorBoard:
```bash
pip install tensorboard
```

Modify the training script to log to TensorBoard, then run:
```bash
tensorboard --logdir=checkpoints/segformer/logs
```

### Key Metrics to Watch

1. **Training Loss**: Should decrease steadily
2. **Validation Loss**: Should decrease, watch for overfitting (gap between train/val)
3. **mIoU (Mean Intersection over Union)**: Primary metric, should increase
4. **Per-class IoU**: Identify which classes are challenging
5. **Dice Coefficient**: Alternative metric, especially good for small objects

---

## 🔧 Troubleshooting

### Common Issues

**1. Out of Memory (even on CPU)**
- Reduce `batch_size` to 1 or 2
- Reduce `input_size` to 256 or 384
- Increase `gradient_accumulation_steps`

**2. Slow Training on CPU**
- Use smaller model variant (SegFormer-B0 instead of B5)
- Reduce `input_size`
- Decrease `num_workers` to 0 or 1
- Consider using mixed precision if you get a GPU later

**3. Poor Convergence**
- Lower learning rate (try 0.00001)
- Increase warmup epochs
- Check data quality and class balance
- Verify mask indices match class mapping

**4. Class Imbalance**
- Add weighted loss in config
- Use oversampling for rare classes
- Adjust augmentation probabilities per class

---

## 🎓 Tips for Best Results

1. **Start Small**: Begin with 256x256 input size and few epochs to verify pipeline
2. **Data Quality**: Ensure masks are accurate - garbage in, garbage out
3. **Augmentation Balance**: Don't over-augment; some defects are subtle
4. **Monitor Per-Class Metrics**: Scratches and contamination might need special attention
5. **Experiment with Models**: Try SegFormer-B1 or B2 if B0 underperforms
6. **Patience**: Transformers may need more epochs than CNNs

---

## 📚 Next Steps

After mastering this pipeline:
1. Experiment with different SegFormer variants (B1-B5)
2. Try ensemble methods with multiple models
3. Implement active learning to prioritize labeling
4. Deploy model with ONNX runtime for faster inference
5. Explore semi-supervised learning for unlabeled data

Good luck with your SWIR board inspection project! 🚀
