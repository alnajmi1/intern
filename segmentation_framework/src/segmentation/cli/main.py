"""
CLI entry point for segmentation framework.
Provides commands for training, evaluation, and inference.
"""

import argparse
import sys
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Segmentation Framework for Industrial Inspection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a segmentation model")
    train_parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/yolo.yaml",
        help="Path to configuration YAML file",
    )
    train_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    eval_parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Data split to evaluate on",
    )
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run inference on images")
    predict_parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )
    predict_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    predict_parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input image or directory",
    )
    predict_parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs/predictions",
        help="Output directory for predictions",
    )
    
    # Convert command (for CVAT export conversion)
    convert_parser = subparsers.add_parser("convert", help="Convert dataset format")
    convert_parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input dataset directory (CVAT export)",
    )
    convert_parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory",
    )
    convert_parser.add_argument(
        "--format",
        type=str,
        default="yolo",
        choices=["yolo", "segformer", "coco"],
        help="Target format",
    )
    
    # Validate config command
    validate_parser = subparsers.add_parser("validate", help="Validate configuration")
    validate_parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.command is None:
        print("Error: No command specified. Use -h for help.")
        sys.exit(1)
    
    if args.command == "train":
        run_training(args)
    elif args.command == "evaluate":
        run_evaluation(args)
    elif args.command == "predict":
        run_prediction(args)
    elif args.command == "convert":
        run_conversion(args)
    elif args.command == "validate":
        run_validation(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


def run_training(args):
    """Run training pipeline."""
    from ..config.config import Config
    from ..config.validator import ConfigValidator
    from ..models.factory import create_model
    from ..datasets.loaders import get_dataloader
    from ..training.trainer import Trainer
    from ..training.loss import get_loss_function
    from ..training.optimizer import get_optimizer, get_scheduler
    from ..utils.device import get_device
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = Config.from_yaml(args.config)
    
    # Validate configuration
    validator = ConfigValidator(config)
    is_valid, warnings, errors = validator.validate()
    validator.print_report()
    
    if not is_valid:
        print("\n❌ Configuration has errors. Please fix them before training.")
        sys.exit(1)
    
    # Setup device
    device = get_device(config.device)
    print(f"\nUsing device: {device}")
    
    # Create model
    print(f"\nCreating model: {config.model.name}")
    model = create_model(
        model_name=config.model.name,
        num_classes=config.num_classes,
        pretrained=config.model.pretrained,
        input_size=config.model.input_size,
    )
    
    # Get data loaders
    print("\nLoading datasets...")
    train_loader = get_dataloader(
        config=config,
        split="train",
        is_training=True,
    )
    val_loader = get_dataloader(
        config=config,
        split="val",
        is_training=False,
    )
    
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    
    # Get loss function
    criterion = get_loss_function(num_classes=config.num_classes)
    
    # Get optimizer and scheduler
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {start_epoch}")
    
    # Start training
    history = trainer.train()
    
    print("\n✅ Training completed!")
    print(f"Best model saved to: {config.experiment.checkpoint_dir}/best.pt")


def run_evaluation(args):
    """Run evaluation pipeline."""
    from ..config.config import Config
    from ..evaluation.evaluator import Evaluator
    
    # Load configuration
    config = Config.from_yaml(args.config)
    
    # Create evaluator
    evaluator = Evaluator(
        config=config,
        checkpoint_path=args.checkpoint,
        split=args.split,
    )
    
    # Run evaluation
    metrics = evaluator.evaluate()
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    print("=" * 60)


def run_prediction(args):
    """Run prediction/inference pipeline."""
    from ..config.config import Config
    from ..inference.predictor import Predictor
    
    # Load configuration
    config = Config.from_yaml(args.config)
    
    # Create predictor
    predictor = Predictor(
        checkpoint_path=args.checkpoint,
        config=config,
    )
    
    # Run prediction
    predictor.predict_directory(
        input_path=args.input,
        output_dir=args.output,
    )
    
    print(f"\n✅ Predictions saved to: {args.output}")


def run_conversion(args):
    """Run dataset conversion pipeline."""
    print(f"Converting dataset from {args.input} to {args.format} format...")
    print("Output directory:", args.output)
    
    # TODO: Implement conversion logic
    print("\n⚠️  Conversion not yet implemented.")
    print("Please refer to the tutorial documentation for manual conversion steps.")


def run_validation(args):
    """Validate configuration file."""
    from ..config.config import Config
    from ..config.validator import ConfigValidator
    
    # Load configuration
    print(f"Validating configuration: {args.config}")
    config = Config.from_yaml(args.config)
    
    # Validate
    validator = ConfigValidator(config)
    is_valid, warnings, errors = validator.validate()
    validator.print_report()
    
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
