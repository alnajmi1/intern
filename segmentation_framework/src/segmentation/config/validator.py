"""
Configuration validator for segmentation framework.
Validates configuration parameters and provides warnings/errors.
"""

from pathlib import Path
from typing import List, Tuple

from ..config.config import Config


class ConfigValidator:
    """Validates configuration parameters."""
    
    VALID_MODEL_NAMES = [
        "yolov8n-seg", "yolov8s-seg", "yolov8m-seg", "yolov8l-seg", "yolov8x-seg",
        "segformer-b0", "segformer-b1", "segformer-b2", "segformer-b3", "segformer-b4", "segformer-b5",
        "sam-vit-b", "sam-vit-l", "sam-vit-h",
    ]
    
    VALID_OPTIMIZERS = ["adamw", "sgd", "adam"]
    VALID_SCHEDULERS = ["cosine", "step", "linear", "none"]
    VALID_BORDER_MODES = ["mirror", "constant", "reflect", "replicate"]
    
    def __init__(self, config: Config):
        self.config = config
        self.warnings: List[str] = []
        self.errors: List[str] = []
    
    def validate(self) -> Tuple[bool, List[str], List[str]]:
        """
        Validate the entire configuration.
        
        Returns:
            Tuple of (is_valid, warnings, errors)
        """
        self.warnings = []
        self.errors = []
        
        self._validate_model()
        self._validate_training()
        self._validate_data()
        self._validate_augmentation()
        self._validate_device()
        
        is_valid = len(self.errors) == 0
        return is_valid, self.warnings, self.errors
    
    def _validate_model(self) -> None:
        """Validate model configuration."""
        model = self.config.model
        
        # Check model name
        if model.name not in self.VALID_MODEL_NAMES:
            self.errors.append(
                f"Invalid model name '{model.name}'. "
                f"Valid options: {', '.join(self.VALID_MODEL_NAMES)}"
            )
        
        # Check input size
        if model.input_size % 32 != 0:
            self.errors.append(f"input_size must be multiple of 32, got {model.input_size}")
        
        # Check num_classes
        if model.num_classes != len(self.config.classes):
            self.warnings.append(
                f"model.num_classes ({model.num_classes}) doesn't match "
                f"number of classes in config ({len(self.config.classes)})"
            )
        
        # SAM-specific checks
        if model.name.startswith("sam"):
            if model.input_size < 1024:
                self.warnings.append(
                    f"SAM models work best with input_size >= 1024, got {model.input_size}"
                )
    
    def _validate_training(self) -> None:
        """Validate training configuration."""
        training = self.config.training
        
        # Check optimizer
        if training.optimizer not in self.VALID_OPTIMIZERS:
            self.errors.append(
                f"Invalid optimizer '{training.optimizer}'. "
                f"Valid options: {', '.join(self.VALID_OPTIMIZERS)}"
            )
        
        # Check scheduler
        if training.scheduler not in self.VALID_SCHEDULERS:
            self.errors.append(
                f"Invalid scheduler '{training.scheduler}'. "
                f"Valid options: {', '.join(self.VALID_SCHEDULERS)}"
            )
        
        # Check learning rate
        if training.learning_rate <= 0:
            self.errors.append(f"learning_rate must be positive, got {training.learning_rate}")
        
        # Check batch size
        if training.batch_size <= 0:
            self.errors.append(f"batch_size must be positive, got {training.batch_size}")
        
        # Check epochs
        if training.epochs <= 0:
            self.errors.append(f"epochs must be positive, got {training.epochs}")
        
        # Learning rate recommendations
        if "segformer" in self.config.model.name.lower():
            if training.learning_rate > 0.0001:
                self.warnings.append(
                    f"SegFormer typically uses lower learning rates (e.g., 0.00005). "
                    f"Current: {training.learning_rate}"
                )
        elif "yolo" in self.config.model.name.lower():
            if training.learning_rate < 0.001 or training.learning_rate > 0.1:
                self.warnings.append(
                    f"YOLO typically uses learning rates between 0.001 and 0.1. "
                    f"Current: {training.learning_rate}"
                )
    
    def _validate_data(self) -> None:
        """Validate data configuration."""
        data = self.config.data
        
        # Check directories exist
        root_path = Path(data.root_dir)
        
        if not root_path.exists():
            self.warnings.append(f"Data root directory does not exist: {root_path}")
        else:
            # Check subdirectories
            for dir_name in [data.raw_dir, data.processed_dir, data.splits_dir]:
                dir_path = root_path / dir_name
                if not dir_path.exists():
                    self.warnings.append(f"Data directory does not exist: {dir_path}")
        
        # Check num_workers
        if data.num_workers < 0:
            self.errors.append(f"num_workers must be non-negative, got {data.num_workers}")
    
    def _validate_augmentation(self) -> None:
        """Validate augmentation configuration."""
        aug = self.config.augmentation
        
        # Check border mode
        if aug.border_mode not in self.VALID_BORDER_MODES:
            self.errors.append(
                f"Invalid border_mode '{aug.border_mode}'. "
                f"Valid options: {', '.join(self.VALID_BORDER_MODES)}"
            )
        
        # Check curriculum_epoch
        if aug.curriculum_epoch <= 0:
            self.errors.append(f"curriculum_epoch must be positive, got {aug.curriculum_epoch}")
        
        # Check static augmentations
        static = aug.static
        if not static.rotations:
            self.warnings.append("No rotations specified in static augmentations")
        
        for flip_type in static.flips:
            if flip_type not in ["x", "y", "xy"]:
                self.errors.append(f"Invalid flip type '{flip_type}'. Valid: x, y, xy")
        
        # Check dynamic augmentations
        dynamic = aug.dynamic
        if dynamic.enabled:
            # Check probability ranges
            prob_params = [
                ("blur_prob", dynamic.blur_prob),
                ("sharpen_prob", dynamic.sharpen_prob),
                ("intensity_prob", dynamic.intensity_prob),
                ("contrast_prob", dynamic.contrast_prob),
                ("gamma_prob", dynamic.gamma_prob),
                ("elastic_prob", dynamic.elastic_prob),
                ("grid_distort_prob", dynamic.grid_distort_prob),
                ("noise_prob", dynamic.noise_prob),
                ("motion_blur_prob", dynamic.motion_blur_prob),
                ("defocus_blur_prob", dynamic.defocus_blur_prob),
                ("cutout_prob", dynamic.cutout_prob),
            ]
            
            for param_name, prob in prob_params:
                if not (0.0 <= prob <= 1.0):
                    self.errors.append(f"{param_name} must be between 0 and 1, got {prob}")
    
    def _validate_device(self) -> None:
        """Validate device configuration."""
        device = self.config.device
        
        # Check seed
        if device.seed < 0:
            self.errors.append(f"seed must be non-negative, got {device.seed}")
        
        # Warnings for CPU training
        if not device.use_cuda:
            model_name = self.config.model.name.lower()
            if "sam" in model_name or "segformer-b" in model_name:
                self.warnings.append(
                    f"Training {self.config.model.name} on CPU will be very slow. "
                    "Consider using a GPU."
                )
    
    def print_report(self) -> None:
        """Print validation report."""
        print("=" * 60)
        print("CONFIGURATION VALIDATION REPORT")
        print("=" * 60)
        
        if self.errors:
            print("\n❌ ERRORS:")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if not self.errors and not self.warnings:
            print("\n✅ Configuration is valid with no warnings!")
        elif not self.errors:
            print("\n✅ Configuration is valid (with warnings)")
        
        print("=" * 60)
