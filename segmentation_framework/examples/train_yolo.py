#!/usr/bin/env python3
"""Example training script for YOLOv8-Seg."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from segmentation import Config, YOLOv8Seg


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8-Seg model")
    parser.add_argument("--config", type=str, default="configs/yolo.yaml")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    
    config = Config.from_yaml(args.config)
    
    if args.data:
        config.data.train = args.data
    if args.epochs:
        config.training.epochs = args.epochs
    
    print(f"Training YOLOv8-Seg: {config.model.name}")
    print(f"Classes: {config.model.num_classes}, Input: {config.model.input_size}")
    print(f"Epochs: {config.training.epochs}, Batch: {config.training.batch_size}")
    
    model = YOLOv8Seg(
        variant=config.model.name,
        num_classes=config.model.num_classes,
        pretrained=config.model.pretrained,
    )
    
    results = model.train_model(
        data_path=config.data.train,
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        imgsz=config.model.input_size,
        device=config.device,
        lr0=config.training.learning_rate,
    )
    
    print(f"Training completed! Results: {results.save_dir}")


if __name__ == "__main__":
    main()
