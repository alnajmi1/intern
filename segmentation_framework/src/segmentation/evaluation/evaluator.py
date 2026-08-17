"""Evaluator class for segmentation models."""
from typing import Dict, Any, Optional, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import SegmentationMetrics
from .visualization import visualize_batch


class Evaluator:
    """Model evaluation handler.
    
    Evaluates a segmentation model on a validation/test dataset and computes
    comprehensive metrics including mIoU, Dice coefficient, and pixel accuracy.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        num_classes: int,
        class_names: Optional[List[str]] = None,
    ):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        
        self.metrics = SegmentationMetrics(num_classes, class_names)
    
    @torch.no_grad()
    def evaluate(
        self,
        data_loader: DataLoader,
        save_visualizations: bool = False,
        viz_save_dir: Optional[str] = None,
        max_viz_samples: int = 8,
    ) -> Dict[str, Any]:
        """Evaluate model on dataset.
        
        Args:
            data_loader: DataLoader for evaluation dataset
            save_visualizations: Whether to save prediction visualizations
            viz_save_dir: Directory to save visualizations
            max_viz_samples: Maximum number of samples to visualize
        
        Returns:
            Dictionary with all computed metrics
        """
        self.model.eval()
        self.metrics.reset()
        
        all_preds = []
        all_targets = []
        all_images = []
        
        pbar = tqdm(data_loader, desc="Evaluating")
        
        for batch in pbar:
            images = batch["image"].to(self.device)
            targets = batch["mask"].numpy()
            
            # Forward pass
            outputs = self.model(images)
            
            # Get predictions (handle different output formats)
            if isinstance(outputs, dict):
                # Some models return dict with 'masks' or 'logits'
                if "masks" in outputs:
                    preds_logits = outputs["masks"]
                elif "logits" in outputs:
                    preds_logits = outputs["logits"]
                else:
                    preds_logits = list(outputs.values())[0]
            else:
                preds_logits = outputs
            
            # Convert logits to class predictions
            if preds_logits.ndim == 4:
                # (B, C, H, W) -> (B, H, W)
                preds = preds_logits.argmax(dim=1).cpu().numpy()
            else:
                preds = preds_logits.cpu().numpy()
            
            # Store for metrics
            self.metrics.update(preds, targets)
            
            # Store for visualization
            all_preds.append(preds)
            all_targets.append(targets)
            all_images.append(images.cpu().numpy())
        
        # Compute final metrics
        results = self.metrics.compute_all()
        
        # Add sample count
        results["num_samples"] = len(data_loader.dataset)
        
        # Save visualizations if requested
        if save_visualizations and viz_save_dir:
            import os
            os.makedirs(viz_save_dir, exist_ok=True)
            
            all_images_np = np.concatenate(all_images, axis=0)
            all_preds_np = np.concatenate(all_preds, axis=0)
            all_targets_np = np.concatenate(all_targets, axis=0)
            
            visualize_batch(
                all_images_np,
                all_preds_np,
                all_targets_np,
                class_names=self.class_names,
                save_dir=viz_save_dir,
                max_samples=max_viz_samples,
            )
        
        return results
    
    def evaluate_and_print(
        self,
        data_loader: DataLoader,
        **kwargs,
    ) -> Dict[str, Any]:
        """Evaluate and print results."""
        results = self.evaluate(data_loader, **kwargs)
        print("\n" + self.metrics.summary())
        return results
