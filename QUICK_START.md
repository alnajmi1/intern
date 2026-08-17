# SWIR Vitrox Board Inspection - Quick Start Guide

## 🎯 Overview

This project provides **two complete segmentation pipelines** for inspecting SWIR Vitrox boards:

1. **SegFormer** - Transformer-based semantic segmentation (higher accuracy, slower)
2. **YOLOv8-Seg** - CNN-based instance segmentation (faster, good accuracy)

Both support:
- ✅ CPU training
- ✅ Curriculum learning with dynamic augmentations
- ✅ Configurable parameters
- ✅ Complete evaluation and visualization
- ✅ Export from CVAT

---

## 📁 Project Structure

```
/workspace/
├── tutorials/
│   ├── segformer_tutorial.md    # Complete SegFormer guide
│   └── yolo_tutorial.md          # Complete YOLOv8 guide
├── scripts/
│   ├── convert_cvat_to_indexed.py    # For SegFormer
│   ├── convert_coco_to_yolo.py       # For YOLO
│   ├── train_segformer.py            # SegFormer training
│   ├── evaluate_segformer.py         # SegFormer evaluation
│   ├── train_yolo.py                 # YOLO training
│   └── evaluate_yolo.py              # YOLO evaluation
├── configs/
│   ├── segformer_config.yaml         # SegFormer configuration
│   └── yolo_config.yaml              # YOLO configuration
└── QUICK_START.md                    # This file
```

---

## 🚀 Quick Start (5 Minutes)

### Option 1: YOLOv8 (Recommended for Beginners)

**Best for**: Fast prototyping, feasibility studies, deployment

```bash
# 1. Create environment
conda create -n yolo_inspection python=3.9 -y
conda activate yolo_inspection
pip install ultralytics opencv-python pillow numpy matplotlib

# 2. Prepare data (after labeling in CVAT)
# Export from CVAT as "COCO 1.0" format
python scripts/convert_coco_to_yolo.py \
    --coco_json data/masks_coco/annotations.json \
    --images_dir data/raw_images \
    --output_dir data/yolo_dataset \
    --classes die scratches contamination other_defects

# 3. Train (CPU, ~2-4 hours for 50 epochs)
python scripts/train_yolo.py \
    --config configs/yolo_config.yaml \
    --epochs 30 \
    --batch_size 2 \
    --model yolov8n-seg.pt

# 4. Evaluate
python scripts/evaluate_yolo.py \
    --config configs/yolo_config.yaml \
    --visualize
```

### Option 2: SegFormer

**Best for**: Maximum accuracy, research, when time allows

```bash
# 1. Create environment
conda create -n segformer_inspection python=3.9 -y
conda activate segformer_inspection
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers albumentations opencv-python pillow numpy

# 2. Prepare data (after labeling in CVAT)
# Export from CVAT as mask format
python scripts/convert_cvat_to_indexed.py \
    --cvat_dir data/cvat_masks \
    --output_dir data/masks_indexed

# 3. Organize data into train/val splits
# (See segformer_tutorial.md for detailed steps)

# 4. Train (CPU, ~1-2 days for 50 epochs)
python scripts/train_segformer.py \
    --config configs/segformer_config.yaml \
    --epochs 30 \
    --batch_size 2 \
    --input_size 512

# 5. Evaluate
python scripts/evaluate_segformer.py \
    --config configs/segformer_config.yaml \
    --checkpoint checkpoints/segformer/best_model.pth \
    --visualize
```

---

## 🏷️ Data Labeling with CVAT

### Step-by-Step

1. **Access CVAT**: Go to cvat.ai or set up locally

2. **Create Project**:
   - Name: "SWIR Board Inspection"
   - Labels: `die`, `scratches`, `contamination`, `other_defects`
   - *Don't label background*

3. **Upload Images**:
   - Upload your SWIR board images
   - Wait for processing

4. **Annotate**:
   - Use **Polygon** tool for precise boundaries
   - For `die`: Trace the entire die boundary
   - For `scratches`: Trace each scratch individually
   - For `contamination`: Trace contaminated regions
   - Save frequently (Ctrl+S)

5. **Export**:
   - For YOLO: Export as "COCO 1.0"
   - For SegFormer: Export as "CVAT for images 1.1" or "Mask RGB"

6. **Convert**: Use the conversion scripts above

---

## 🎨 Augmentation Strategy

Both models use **Curriculum Learning**:

### Static Augmentations (Every Epoch)
- ✅ Rotate: 0°, 90°, 180°, 270°
- ✅ Flip: Horizontal, Vertical, Both
- ✅ Random crop (90% of image)
- ✅ Color jitter (HSV)

### Dynamic Augmentations (Every N Epochs, default N=2)
- ✅ Small XY shift (±10 pixels)
- ✅ Small rotation (±5°)
- ✅ Gaussian blur (σ: 0.5-1.5)
- ✅ Sharpening (unsharp mask)
- ✅ Intensity offsets (±15)
- ✅ Contrast scaling (0.85-1.15)
- ✅ Gamma correction (0.9-1.1)
- ✅ Elastic transform
- ✅ Grid distortion
- ✅ Gaussian noise
- ✅ Speckle noise
- ✅ Motion blur (optional)
- ✅ Defocus blur (optional)
- ✅ Cutout/Coarse dropout

**Border Mode**: Mirror/reflect for all geometric transforms

---

## ⚙️ Configuration Highlights

### Key Parameters to Tune

| Parameter | YOLO Default | SegFormer Default | When to Change |
|-----------|-------------|-------------------|----------------|
| **Learning Rate** | 0.01 | 0.00005 | If not converging |
| **Batch Size** | 4 (CPU: 2) | 4 (CPU: 2) | Reduce if OOM |
| **Input Size** | 640 | 640 | Increase for small defects |
| **Epochs** | 50 | 50 | Increase if underfitting |
| **Model Variant** | yolov8n-seg | segformer-b0 | Upgrade for better accuracy |

### Model Variants

**YOLOv8:**
- `yolov8n-seg.pt` - Nano (fastest, ~3M params)
- `yolov8s-seg.pt` - Small (balanced, ~11M params)
- `yolov8m-seg.pt` - Medium (accurate, ~27M params)
- `yolov8l-seg.pt` - Large (very accurate, ~47M params)

**SegFormer:**
- `segformer-b0` - Smallest (fastest)
- `segformer-b1` to `b5` - Progressively larger/more accurate

---

## 📊 Evaluation Metrics

### YOLOv8 Metrics
- **mAP50**: Mean Average Precision at IoU=0.50
- **mAP50-95**: Average mAP across IoU thresholds 0.50-0.95
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)

### SegFormer Metrics
- **mIoU**: Mean Intersection over Union
- **Pixel Accuracy**: Correct pixels / Total pixels
- **Dice Coefficient**: 2×TP / (2×TP + FP + FN)
- **Per-class IoU**: IoU for each defect type

### Good Benchmarks (Target)
| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| mAP50/mIoU | 0.50 | 0.70 | 0.85+ |
| Precision | 0.60 | 0.80 | 0.90+ |
| Recall | 0.60 | 0.75 | 0.85+ |

---

## 🔧 Troubleshooting Quick Reference

### Problem: Training Too Slow on CPU
**Solution:**
```yaml
# Use smaller model
model: yolov8n-seg.pt  # or segformer-b0

# Reduce input size
input_size: 320  # or 256

# Reduce batch size
batch_size: 1
```

### Problem: Out of Memory
**Solution:**
```yaml
batch_size: 1
input_size: 256
cache: false  # Don't cache images
multi_scale: false
```

### Problem: Poor Detection of Small Scratches
**Solution:**
```yaml
# Increase resolution
input_size: 896  # or 1024

# Use larger model
model: yolov8s-seg.pt  # or segformer-b1

# Adjust loss weights (YOLO)
box_gain: 10.0
dfl_gain: 2.0
```

### Problem: Overfitting
**Solution:**
```yaml
# Increase regularization
weight_decay: 0.001
label_smoothing: 0.1

# More augmentation
augmentation:
  mosaic: 1.0
  mixup: 0.1
```

### Problem: Class Imbalance
**Solution:**
1. Collect more data for rare classes
2. Oversample images with rare defects
3. Use weighted loss (modify training script)
4. Focus augmentation on rare classes

---

## 📈 Training Timeline (CPU Estimates)

| Model | Input Size | Batch Size | Epochs | Time (CPU) | Time (GPU) |
|-------|-----------|------------|--------|------------|------------|
| YOLOv8-n | 640 | 4 | 50 | ~3-4 hours | ~15 min |
| YOLOv8-s | 640 | 4 | 50 | ~6-8 hours | ~30 min |
| YOLOv8-m | 640 | 2 | 50 | ~12-16 hours | ~1 hour |
| SegFormer-B0 | 512 | 4 | 50 | ~1-2 days | ~2-3 hours |
| SegFormer-B1 | 512 | 2 | 50 | ~3-4 days | ~5-6 hours |

*Estimates based on modern CPU (8 cores), actual times vary*

---

## 🎯 Recommended Workflow

### Week 1: Data Collection & Labeling
- [ ] Collect 100-500 SWIR board images
- [ ] Set up CVAT account
- [ ] Label 20-30 images for initial training
- [ ] Export and convert data

### Week 2: Initial Training (YOLO)
- [ ] Train YOLOv8-nano for 30 epochs
- [ ] Evaluate results
- [ ] Identify weak classes
- [ ] Label more images for weak classes

### Week 3: Iteration & Improvement
- [ ] Retrain with more data
- [ ] Try YOLOv8-small
- [ ] Experiment with augmentations
- [ ] Fine-tune hyperparameters

### Week 4: Advanced (Optional)
- [ ] Try SegFormer for comparison
- [ ] Ensemble models
- [ ] Deploy best model
- [ ] Set up inference pipeline

---

## 💡 Pro Tips

1. **Start Small**: Begin with 50 images and YOLOv8-nano
2. **Label Quality > Quantity**: 100 well-labeled images beat 500 poorly labeled ones
3. **Monitor Per-Class Metrics**: Catch weak classes early
4. **Use Validation Set**: Always keep 20% for validation
5. **Save Checkpoints**: Models can improve unexpectedly
6. **Visualize Predictions**: Don't just trust numbers
7. **Iterate**: First model won't be perfect - that's normal!

---

## 📞 Getting Help

### Common Questions

**Q: How many images do I need?**
A: Start with 50-100, aim for 200-500 for production quality

**Q: Which model should I choose?**
A: Start with YOLOv8-nano for speed, upgrade if needed

**Q: How long should I train?**
A: 30-50 epochs usually sufficient, watch validation metrics

**Q: My model isn't learning, what's wrong?**
A: Check: learning rate, data quality, class balance, augmentations

**Q: Should I use GPU?**
A: Nice to have, but not required. CPU works fine for experimentation

---

## 🚀 Next Steps After Feasibility

Once you've proven the concept:

1. **Scale Up**: More data, larger models
2. **Optimize**: Hyperparameter tuning, ensemble methods
3. **Deploy**: ONNX/TensorRT export, integration
4. **Monitor**: Track performance in production
5. **Improve**: Active learning, continuous labeling

---

## 📚 Resources

- **SegFormer Tutorial**: `/workspace/tutorials/segformer_tutorial.md`
- **YOLO Tutorial**: `/workspace/tutorials/yolo_tutorial.md`
- **Config Files**: `/workspace/configs/`
- **Scripts**: `/workspace/scripts/`

Good luck with your SWIR board inspection project! 🎉
