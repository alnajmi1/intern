"""
Loss functions for segmentation training.
Supports CrossEntropy, Dice Loss, Focal Loss, and combinations.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            logits: Predicted logits (B, C, H, W)
            targets: Target masks (B, H, W) with class indices
        
        Returns:
            Dice loss value
        """
        # Convert targets to one-hot
        num_classes = logits.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # Apply softmax to logits
        probs = F.softmax(logits, dim=1)
        
        # Compute Dice coefficient for each class
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return mean Dice loss across classes
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(
        self, 
        alpha: float = 0.25, 
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal loss.
        
        Args:
            logits: Predicted logits (B, C, H, W)
            targets: Target masks (B, H, W) with class indices
        
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        
        # Compute focal weight
        pt = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1.0 - pt) ** self.gamma
        
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """Combined loss (CrossEntropy + Dice)."""
    
    def __init__(
        self, 
        ce_weight: float = 0.5, 
        dice_weight: float = 0.5,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.num_classes = num_classes
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            logits: Predicted logits (B, C, H, W)
            targets: Target masks (B, H, W) with class indices
        
        Returns:
            Combined loss value
        """
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        
        return self.ce_weight * ce + self.dice_weight * dice


def get_loss_function(num_classes: int, loss_type: str = "combined") -> nn.Module:
    """
    Get loss function based on type.
    
    Args:
        num_classes: Number of segmentation classes
        loss_type: Type of loss ("crossentropy", "dice", "focal", "combined")
    
    Returns:
        Loss function module
    """
    loss_type = loss_type.lower()
    
    if loss_type == "crossentropy":
        return nn.CrossEntropyLoss()
    elif loss_type == "dice":
        return DiceLoss()
    elif loss_type == "focal":
        return FocalLoss()
    elif loss_type == "combined":
        return CombinedLoss(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
