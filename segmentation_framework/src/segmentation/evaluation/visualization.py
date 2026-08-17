"""Visualization utilities for segmentation results."""
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import cv2


# Default color map for classes (RGB)
DEFAULT_COLORS = [
    [0, 0, 0],       # background - black
    [255, 0, 0],     # die - red
    [0, 255, 0],     # scratches - green
    [0, 0, 255],     # contamination - blue
    [255, 255, 0],   # other_defects - cyan
    [255, 0, 255],   # additional classes - magenta
]


def create_color_mask(
    mask: np.ndarray,
    colors: Optional[List[List[int]]] = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """Convert index mask to RGB color mask.
    
    Args:
        mask: Index mask (H, W) with class indices
        colors: List of RGB colors per class
        alpha: Transparency for overlay
    
    Returns:
        Color mask (H, W, 3) in uint8
    """
    if colors is None:
        colors = DEFAULT_COLORS
    
    num_classes = len(colors)
    h, w = mask.shape
    
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for cls_idx in range(num_classes):
        class_mask = (mask == cls_idx)
        color = colors[cls_idx % len(colors)]
        color_mask[class_mask] = color
    
    return color_mask


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    colors: Optional[List[List[int]]] = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay colored mask on original image.
    
    Args:
        image: Original image (H, W, 3) in uint8 or float [0,1]
        mask: Index mask (H, W)
        colors: List of RGB colors per class
        alpha: Transparency (0=only image, 1=only mask)
    
    Returns:
        Overlay image (H, W, 3) in uint8
    """
    # Ensure image is uint8
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    
    color_mask = create_color_mask(mask, colors)
    
    # Blend image and mask
    overlay = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)
    
    return overlay


def visualize_prediction(
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """Create visualization of prediction vs ground truth.
    
    Args:
        image: Input image (H, W, 3)
        pred_mask: Predicted mask (H, W)
        gt_mask: Ground truth mask (H, W), optional
        class_names: List of class names for legend
        save_path: Path to save figure, optional
        show: Whether to display plot
    
    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(DEFAULT_COLORS))]
    
    fig, axes = plt.subplots(1, 3 if gt_mask is not None else 2, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Prediction
    pred_overlay = overlay_mask_on_image(image, pred_mask)
    axes[1].imshow(pred_overlay)
    axes[1].set_title("Prediction")
    axes[1].axis("off")
    
    # Ground truth (if provided)
    if gt_mask is not None:
        gt_overlay = overlay_mask_on_image(image, gt_mask)
        axes[2].imshow(gt_overlay)
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")
    
    # Add legend
    legend_elements = []
    for i, name in enumerate(class_names):
        color = tuple(c / 255.0 for c in DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, label=name))
    
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(class_names),
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def visualize_batch(
    images: np.ndarray,
    preds: np.ndarray,
    targets: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
    max_samples: int = 8,
) -> List[plt.Figure]:
    """Visualize a batch of predictions.
    
    Args:
        images: Batch of images (B, H, W, 3) or (B, 3, H, W)
        preds: Batch of predicted masks (B, H, W)
        targets: Batch of ground truth masks (B, H, W), optional
        class_names: List of class names
        save_dir: Directory to save visualizations
        max_samples: Maximum number of samples to visualize
    
    Returns:
        List of figures
    """
    if images.ndim == 4 and images.shape[1] == 3:
        # Convert from (B, C, H, W) to (B, H, W, C)
        images = np.transpose(images, (0, 2, 3, 1))
    
    num_samples = min(len(images), max_samples)
    figures = []
    
    for i in range(num_samples):
        img = images[i]
        pred = preds[i]
        target = targets[i] if targets is not None else None
        
        save_path = None
        if save_dir:
            save_path = f"{save_dir}/sample_{i:03d}.png"
        
        fig = visualize_prediction(
            img, pred, target, class_names, save_path=save_path
        )
        figures.append(fig)
        plt.close(fig)
    
    return figures
