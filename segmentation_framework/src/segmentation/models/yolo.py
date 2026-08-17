"""YOLOv8 segmentation model wrapper."""
from typing import Any, Dict, Optional
import torch
import torch.nn as nn


class YOLOv8Seg(nn.Module):
    """YOLOv8 Segmentation model wrapper.
    
    Supports all YOLOv8 segmentation variants:
    - yolov8n-seg (nano)
    - yolov8s-seg (small)
    - yolov8m-seg (medium)
    - yolov8l-seg (large)
    - yolov8x-seg (extra large)
    """
    
    def __init__(
        self,
        variant: str = "yolov8n-seg",
        num_classes: int = 5,
        pretrained: bool = True,
        input_size: int = 640,
    ):
        super().__init__()
        
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "Please install ultralytics: pip install ultralytics"
            )
        
        self.variant = variant
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Load YOLOv8 model
        if pretrained:
            self.model = YOLO(f"{variant}.pt")
        else:
            self.model = YOLO(f"{variant}.yaml")
        
        # Note: Ultralytics YOLO handles class mapping internally
        # We'll use their training API rather than direct PyTorch
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Note: For training, use the Ultralytics .train() method directly.
        This forward method is for inference only.
        """
        # Convert tensor to numpy for Ultralytics
        # This is a simplified version - in practice, use .predict() or .train()
        raise NotImplementedError(
            "For YOLOv8, use the Ultralytics .train() method directly. "
            "See tutorials/yolo_tutorial.md for usage."
        )
    
    def train_model(
        self,
        data_path: str,
        epochs: int = 100,
        batch_size: int = 8,
        imgsz: int = 640,
        device: str = "cpu",
        **kwargs: Any,
    ):
        """Train YOLOv8 model using Ultralytics API.
        
        Args:
            data_path: Path to dataset (YAML format)
            epochs: Number of training epochs
            batch_size: Batch size
            imgsz: Image size
            device: Device ('cpu', 'cuda', 'cuda:0', etc.)
            **kwargs: Additional training arguments
        
        Returns:
            Training results
        """
        results = self.model.train(
            data=data_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=device,
            classes=self.num_classes,
            **kwargs,
        )
        
        return results
