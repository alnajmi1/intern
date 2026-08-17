"""Data loaders and split utilities."""
from typing import Optional, Dict, Any, Tuple
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import torch


def create_data_loader(
    dataset: Dataset,
    batch_size: int = 8,
    num_workers: int = 0,  # 0 for CPU, >0 for GPU
    shuffle: bool = True,
    pin_memory: bool = False,
) -> DataLoader:
    """Create a PyTorch DataLoader with optimal settings for CPU/GPU."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True if len(dataset) > batch_size else False,
    )


def train_val_split(
    dataset: Dataset,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """Split dataset into training and validation sets."""
    generator = torch.Generator().manual_seed(seed)
    
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )
    
    return train_dataset, val_dataset


def train_val_test_split(
    dataset: Dataset,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Split dataset into training, validation, and test sets."""
    generator = torch.Generator().manual_seed(seed)
    
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    test_size = int(total_size * test_ratio)
    train_size = total_size - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )
    
    return train_dataset, val_dataset, test_dataset
