#!/usr/bin/env python3
"""Example training script for SegFormer."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from segmentation import Config, SegFormer, IndexedMaskDataset, create_data_loader, train_val_split


def main():
    parser = argparse.ArgumentParser(description="Train SegFormer model")
    parser.add_argument("--config", type=str, default="configs/segformer.yaml")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    
    config = Config.from_yaml(args.config)
    
    if args.data:
        config.data.train = args.data
    if args.epochs:
        config.training.epochs = args.epochs
    
    print(f"Training SegFormer: {config.model.name}")
    print(f"Classes: {config.model.num_classes}, Input: {config.model.input_size}")
    print(f"Epochs: {config.training.epochs}, Batch: {config.training.batch_size}")
    
    # Create dataset
    dataset = IndexedMaskDataset(
        root_dir=config.data.train,
        input_size=(config.model.input_size, config.model.input_size),
        classes=config.data.classes,
    )
    
    # Split into train/val
    train_dataset, val_dataset = train_val_split(
        dataset, 
        val_ratio=0.2,
        seed=42
    )
    
    # Create model
    model = SegFormer(
        variant=config.model.name,
        num_classes=config.model.num_classes,
        pretrained=config.model.pretrained,
        input_size=(config.model.input_size, config.model.input_size),
    )
    
    # Train using Hugging Face Trainer
    output_dir = f"outputs/experiments/segformer_{config.model.name}"
    
    trainer = model.train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=output_dir,
        num_train_epochs=config.training.epochs,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        device=config.device,
    )
    
    print(f"Training completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
