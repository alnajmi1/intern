"""
Device utility for segmentation framework.
Handles CPU/GPU device selection and configuration.
"""

import random

import numpy as np
import torch


def get_device(device_config) -> torch.device:
    """
    Get PyTorch device based on configuration.
    
    Args:
        device_config: Device configuration object
    
    Returns:
        torch.device object
    """
    if device_config.use_cuda and torch.cuda.is_available():
        device = torch.device(f"cuda:{device_config.cuda_device}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        if device_config.use_cuda and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
        else:
            print("Using CPU")
    
    # Set deterministic mode if requested
    if device_config.deterministic:
        set_deterministic(device_config.seed)
    
    return device


def set_deterministic(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Enable deterministic algorithms (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def check_memory_usage() -> dict:
    """
    Check current memory usage.
    
    Returns:
        Dictionary with memory statistics
    """
    stats = {
        "cpu_allocated": 0,
        "cpu_cached": 0,
        "gpu_allocated": 0,
        "gpu_cached": 0,
    }
    
    if torch.cuda.is_available():
        stats["gpu_allocated"] = torch.cuda.memory_allocated() / 1024**2  # MB
        stats["gpu_cached"] = torch.cuda.memory_reserved() / 1024**2  # MB
        stats["gpu_max_allocated"] = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    return stats


def print_memory_summary() -> None:
    """Print memory usage summary."""
    stats = check_memory_usage()
    
    print("\nMemory Usage:")
    print(f"  GPU Allocated: {stats['gpu_allocated']:.2f} MB")
    print(f"  GPU Cached: {stats['gpu_cached']:.2f} MB")
    
    if stats.get('gpu_max_allocated'):
        print(f"  GPU Max Allocated: {stats['gpu_max_allocated']:.2f} MB")
