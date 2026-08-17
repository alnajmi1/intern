"""
Model factory for creating segmentation models.
Supports YOLOv8-Seg, SegFormer, and SAM architectures.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def create_model(
    model_name: str,
    num_classes: int = 5,
    pretrained: bool = True,
    input_size: int = 640,
    **kwargs: Any,
) -> nn.Module:
    """
    Create a segmentation model based on name.
    
    Args:
        model_name: Model identifier (e.g., "yolov8n-seg", "segformer-b0")
        num_classes: Number of segmentation classes
        pretrained: Whether to use pretrained weights
        input_size: Input image size
        **kwargs: Additional model-specific arguments
    
    Returns:
        PyTorch model
    
    Raises:
        ValueError: If model_name is not recognized
    """
    model_name_lower = model_name.lower()
    
    # YOLOv8-Seg models
    if model_name_lower.startswith("yolov8"):
        return _create_yolo_model(model_name_lower, num_classes, pretrained)
    
    # SegFormer models
    elif model_name_lower.startswith("segformer"):
        return _create_segformer_model(model_name_lower, num_classes, pretrained)
    
    # SAM models
    elif model_name_lower.startswith("sam"):
        return _create_sam_model(model_name_lower, pretrained)
    
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported: yolov8*-seg, segformer-b*, sam-vit-*"
        )


def _create_yolo_model(
    model_name: str,
    num_classes: int,
    pretrained: bool,
) -> nn.Module:
    """Create YOLOv8-Seg model."""
    try:
        from ultralytics import YOLO
        
        # Map model names to YOLO variants
        model_map = {
            "yolov8n-seg": "yolov8n-seg.pt",
            "yolov8s-seg": "yolov8s-seg.pt",
            "yolov8m-seg": "yolov8m-seg.pt",
            "yolov8l-seg": "yolov8l-seg.pt",
            "yolov8x-seg": "yolov8x-seg.pt",
        }
        
        yolo_path = model_map.get(model_name)
        if yolo_path is None:
            raise ValueError(f"Unknown YOLO variant: {model_name}")
        
        # Load pretrained model
        if pretrained:
            model = YOLO(yolo_path)
        else:
            model = YOLO()
        
        # Modify number of classes if needed
        # Note: YOLO handles this internally during training
        return model
    
    except ImportError:
        raise ImportError(
            "ultralytics package required for YOLO models. "
            "Install with: pip install ultralytics"
        )


def _create_segformer_model(
    model_name: str,
    num_classes: int,
    pretrained: bool,
) -> nn.Module:
    """Create SegFormer model."""
    try:
        from transformers import AutoModelForSemanticSegmentation
        
        # Map model names to HuggingFace model IDs
        model_map = {
            "segformer-b0": "nvidia/segformer-b0-finetuned-ade-512-512",
            "segformer-b1": "nvidia/segformer-b1-finetuned-ade-512-512",
            "segformer-b2": "nvidia/segformer-b2-finetuned-ade-512-512",
            "segformer-b3": "nvidia/segformer-b3-finetuned-ade-512-512",
            "segformer-b4": "nvidia/segformer-b4-finetuned-ade-512-512",
            "segformer-b5": "nvidia/segformer-b5-finetuned-ade-512-512",
        }
        
        model_id = model_map.get(model_name)
        if model_id is None:
            raise ValueError(f"Unknown SegFormer variant: {model_name}")
        
        # Load model
        if pretrained:
            model = AutoModelForSemanticSegmentation.from_pretrained(
                model_id,
                ignore_mismatched_sizes=True,  # Allow different num_classes
            )
        else:
            # For non-pretrained, we'd need to initialize from scratch
            # This is more complex - using pretrained by default
            model = AutoModelForSemanticSegmentation.from_pretrained(
                model_id,
                ignore_mismatched_sizes=True,
            )
        
        # Update classifier for our number of classes
        if hasattr(model, 'classifier'):
            in_channels = model.classifier.in_channels
            model.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        
        return model
    
    except ImportError:
        raise ImportError(
            "transformers package required for SegFormer models. "
            "Install with: pip install transformers"
        )


def _create_sam_model(
    model_name: str,
    pretrained: bool,
) -> nn.Module:
    """Create SAM model."""
    try:
        from segment_anything import sam_model_registry
        
        # Map model names to SAM checkpoints
        model_map = {
            "sam-vit-b": ("vit_b", "sam_vit_b_01ec64.pth"),
            "sam-vit-l": ("vit_l", "sam_vit_l_0b3195.pth"),
            "sam-vit-h": ("vit_h", "sam_vit_h_4b8939.pth"),
        }
        
        if model_name not in model_map:
            raise ValueError(f"Unknown SAM variant: {model_name}")
        
        model_type, checkpoint_url = model_map[model_name]
        
        # Note: SAM requires manual download of checkpoints
        # For now, return a placeholder - actual implementation needs checkpoint path
        raise NotImplementedError(
            "SAM model requires manual checkpoint download. "
            "See https://github.com/facebookresearch/segment-anything"
        )
    
    except ImportError:
        raise ImportError(
            "segment-anything package required for SAM models. "
            "Install with: pip install segment-anything"
        )


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a model architecture.
    
    Args:
        model_name: Model identifier
    
    Returns:
        Dictionary with model information
    """
    model_info = {
        # YOLOv8-Seg
        "yolov8n-seg": {
            "params": "3.2M",
            "speed": "Fast",
            "accuracy": "Good",
            "recommended_for": "CPU training, quick prototyping",
        },
        "yolov8s-seg": {
            "params": "11.8M",
            "speed": "Fast",
            "accuracy": "Better",
            "recommended_for": "Balanced speed/accuracy",
        },
        "yolov8m-seg": {
            "params": "27.3M",
            "speed": "Medium",
            "accuracy": "Good",
            "recommended_for": "Production use",
        },
        # SegFormer
        "segformer-b0": {
            "params": "3.8M",
            "speed": "Medium",
            "accuracy": "Good",
            "recommended_for": "CPU training, smaller datasets",
        },
        "segformer-b1": {
            "params": "13.9M",
            "speed": "Medium",
            "accuracy": "Better",
            "recommended_for": "Balanced performance",
        },
        "segformer-b2": {
            "params": "25.3M",
            "speed": "Slow",
            "accuracy": "Very Good",
            "recommended_for": "GPU training, high accuracy",
        },
        # SAM
        "sam-vit-b": {
            "params": "90.7M",
            "speed": "Slow",
            "accuracy": "Excellent",
            "recommended_for": "Few-shot, interactive segmentation",
        },
    }
    
    return model_info.get(model_name.lower(), {})
