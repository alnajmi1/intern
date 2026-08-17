"""Evaluation metrics for segmentation models."""
from typing import Dict, Any, Optional
import numpy as np
import torch


def compute_iou(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = -100,
) -> np.ndarray:
    """Compute Intersection over Union (IoU) for each class.
    
    Args:
        pred: Predicted masks (H, W) or (B, H, W)
        target: Ground truth masks (H, W) or (B, H, W)
        num_classes: Number of classes
        ignore_index: Index to ignore (e.g., padding)
    
    Returns:
        IoU per class (num_classes,)
    """
    if pred.ndim == 3:
        # Batch mode
        ious = []
        for b in range(pred.shape[0]):
            ious.append(compute_iou(pred[b], target[b], num_classes, ignore_index))
        return np.mean(ious, axis=0)
    
    ious = np.zeros(num_classes)
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()
        
        if union > 0:
            ious[cls] = intersection / union
        else:
            ious[cls] = np.nan
    
    return ious


def compute_dice(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Compute Dice coefficient for each class.
    
    Args:
        pred: Predicted masks
        target: Ground truth masks
        num_classes: Number of classes
        epsilon: Small value to avoid division by zero
    
    Returns:
        Dice coefficient per class (num_classes,)
    """
    if pred.ndim == 3:
        dice_scores = []
        for b in range(pred.shape[0]):
            dice_scores.append(compute_dice(pred[b], target[b], num_classes, epsilon))
        return np.mean(dice_scores, axis=0)
    
    dice = np.zeros(num_classes)
    
    for cls in range(num_classes):
        pred_cls = (pred == cls).astype(np.float32)
        target_cls = (target == cls).astype(np.float32)
        
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        
        if union > 0:
            dice[cls] = (2.0 * intersection + epsilon) / (union + epsilon)
        else:
            dice[cls] = np.nan
    
    return dice


def compute_pixel_accuracy(
    pred: np.ndarray,
    target: np.ndarray,
    ignore_index: int = -100,
) -> float:
    """Compute pixel-wise accuracy.
    
    Returns:
        Pixel accuracy (scalar)
    """
    if pred.ndim == 3:
        accs = []
        for b in range(pred.shape[0]):
            accs.append(compute_pixel_accuracy(pred[b], target[b], ignore_index))
        return np.mean(accs)
    
    valid = target != ignore_index
    pred_valid = pred[valid]
    target_valid = target[valid]
    
    if len(target_valid) == 0:
        return np.nan
    
    accuracy = (pred_valid == target_valid).sum() / len(target_valid)
    return accuracy


class SegmentationMetrics:
    """Comprehensive segmentation metrics calculator."""
    
    def __init__(self, num_classes: int, class_names: Optional[list] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.all_preds = []
        self.all_targets = []
    
    def update(self, pred: np.ndarray, target: np.ndarray):
        """Add new predictions and targets to metrics."""
        self.all_preds.append(pred)
        self.all_targets.append(target)
    
    def compute_all(self) -> Dict[str, Any]:
        """Compute all metrics on accumulated data."""
        if len(self.all_preds) == 0:
            return {}
        
        all_preds = np.concatenate(self.all_preds, axis=0)
        all_targets = np.concatenate(self.all_targets, axis=0)
        
        iou_per_class = compute_iou(all_preds, all_targets, self.num_classes)
        dice_per_class = compute_dice(all_preds, all_targets, self.num_classes)
        pixel_acc = compute_pixel_accuracy(all_preds, all_targets)
        
        # Mean IoU (excluding NaN)
        miou = np.nanmean(iou_per_class)
        
        # Mean Dice
        mdice = np.nanmean(dice_per_class)
        
        results = {
            "mIoU": miou,
            "pixel_accuracy": pixel_acc,
            "mean_dice": mdice,
            "iou_per_class": iou_per_class,
            "dice_per_class": dice_per_class,
        }
        
        # Add per-class metrics with names
        for i, name in enumerate(self.class_names):
            results[f"{name}_iou"] = iou_per_class[i]
            results[f"{name}_dice"] = dice_per_class[i]
        
        return results
    
    def summary(self) -> str:
        """Return a formatted string summary of metrics."""
        metrics = self.compute_all()
        
        if not metrics:
            return "No metrics computed yet."
        
        lines = [
            "=" * 50,
            "SEGMENTATION METRICS SUMMARY",
            "=" * 50,
            f"Mean IoU:      {metrics['mIoU']:.4f}",
            f"Pixel Acc:     {metrics['pixel_accuracy']:.4f}",
            f"Mean Dice:     {metrics['mean_dice']:.4f}",
            "-" * 50,
            "Per-Class IoU:",
        ]
        
        for i, name in enumerate(self.class_names):
            iou_val = metrics['iou_per_class'][i]
            dice_val = metrics['dice_per_class'][i]
            lines.append(f"  {name:20s}: IoU={iou_val:.4f}, Dice={dice_val:.4f}")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)
