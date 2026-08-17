"""Learning rate schedulers."""
from typing import Optional, Any
import torch
from torch.optim import Optimizer


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "cosine",
    num_epochs: int = 100,
    warmup_epochs: int = 5,
    step_size: int = 30,
    gamma: float = 0.1,
    min_lr: float = 1e-6,
) -> Any:
    """Create learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler ('cosine', 'step', 'linear', 'none')
        num_epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs
        step_size: Step size for StepLR
        gamma: Gamma for StepLR
        min_lr: Minimum learning rate
    
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=min_lr,
        )
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
    elif scheduler_type == "linear":
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                return max(0, 1 - (epoch - warmup_epochs) / (num_epochs - warmup_epochs))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_type == "none" or scheduler_type is None:
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler
