"""
Optimizer and scheduler utilities for segmentation training.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, StepLR, LinearLR


def get_optimizer(model: nn.Module, config) -> Optimizer:
    """
    Create optimizer based on configuration.
    
    Args:
        model: PyTorch model
        config: Configuration object with training parameters
    
    Returns:
        Optimizer instance
    """
    optimizer_type = config.training.optimizer.lower()
    lr = config.training.learning_rate
    weight_decay = config.training.weight_decay
    momentum = config.training.momentum
    
    if optimizer_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_type == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_type == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def get_scheduler(
    optimizer: Optimizer, 
    config,
    num_training_steps: Optional[int] = None,
) -> Optional[_LRScheduler]:
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: Optimizer instance
        config: Configuration object with training parameters
        num_training_steps: Total number of training steps (optional)
    
    Returns:
        Learning rate scheduler or None if scheduler type is "none"
    """
    scheduler_type = config.training.scheduler.lower()
    
    if scheduler_type == "none":
        return None
    
    epochs = config.training.epochs
    
    if scheduler_type == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=1e-6,
        )
    elif scheduler_type == "step":
        # Step decay every 30 epochs by factor of 0.1
        return StepLR(
            optimizer,
            step_size=30,
            gamma=0.1,
        )
    elif scheduler_type == "linear":
        # Linear warmup + linear decay
        warmup_epochs = config.training.warmup_epochs
        
        # Use LinearLR for simple linear decay
        return LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=epochs,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def get_warmup_scheduler(
    optimizer: Optimizer,
    warmup_epochs: int,
    warmup_lr_start: float = 0.0,
) -> _LRScheduler:
    """
    Create warmup scheduler.
    
    Args:
        optimizer: Optimizer instance
        warmup_epochs: Number of warmup epochs
        warmup_lr_start: Starting learning rate for warmup
    
    Returns:
        Linear warmup scheduler
    """
    return LinearLR(
        optimizer,
        start_factor=warmup_lr_start,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )


def get_optimizer_info(optimizer_type: str) -> dict:
    """
    Get information about an optimizer type.
    
    Args:
        optimizer_type: Optimizer name
    
    Returns:
        Dictionary with optimizer information
    """
    info = {
        "adamw": {
            "description": "Adam with decoupled weight decay",
            "recommended_for": "Transformers, SegFormer, most modern architectures",
            "typical_lr": "1e-4 to 1e-3",
            "typical_weight_decay": "0.01 to 0.1",
        },
        "adam": {
            "description": "Adaptive Moment Estimation",
            "recommended_for": "General purpose, CNNs",
            "typical_lr": "1e-3 to 1e-2",
            "typical_weight_decay": "1e-4 to 1e-3",
        },
        "sgd": {
            "description": "Stochastic Gradient Descent with momentum",
            "recommended_for": "When sharp minima are a concern, some CNNs",
            "typical_lr": "0.01 to 0.1",
            "typical_momentum": "0.9 to 0.99",
        },
    }
    
    return info.get(optimizer_type.lower(), {})
