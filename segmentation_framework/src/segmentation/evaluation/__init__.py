"""Evaluation module initialization."""
from .metrics import (
    compute_iou,
    compute_dice,
    compute_pixel_accuracy,
    SegmentationMetrics,
)
from .visualization import (
    create_color_mask,
    overlay_mask_on_image,
    visualize_prediction,
    visualize_batch,
)
from .evaluator import Evaluator

__all__ = [
    "compute_iou",
    "compute_dice",
    "compute_pixel_accuracy",
    "SegmentationMetrics",
    "create_color_mask",
    "overlay_mask_on_image",
    "visualize_prediction",
    "visualize_batch",
    "Evaluator",
]