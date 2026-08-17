"""
Configuration management for segmentation framework.
Handles loading, validation, and merging of configuration files.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, validator


class ModelConfig(BaseModel):
    """Model configuration."""
    name: str = "yolov8n-seg"
    num_classes: int = 5
    pretrained: bool = True
    input_size: int = 640
    
    @validator('input_size')
    def check_input_size(cls, v):
        if v % 32 != 0:
            raise ValueError(f"input_size must be multiple of 32, got {v}")
        return v


class TrainingConfig(BaseModel):
    """Training configuration."""
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 0.01
    weight_decay: float = 0.0005
    momentum: float = 0.937
    warmup_epochs: int = 3
    warmup_lr_start: float = 0.0
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    patience: int = 20
    gradient_clip: float = 1.0
    mixed_precision: bool = False


class DataConfig(BaseModel):
    """Data configuration."""
    root_dir: str = "data"
    raw_dir: str = "raw"
    processed_dir: str = "processed"
    splits_dir: str = "splits"
    train_split: str = "train.txt"
    val_split: str = "val.txt"
    test_split: str = "test.txt"
    num_workers: int = 4
    pin_memory: bool = False


class StaticAugmentationConfig(BaseModel):
    """Static augmentation configuration."""
    rotations: List[int] = [0, 90, 180, 270]
    flips: List[str] = ["x", "y", "xy"]
    random_crop: bool = True
    crop_size_ratio: float = 0.9
    prob_rotation: float = 0.5
    prob_flip: float = 0.5
    prob_crop: float = 0.3


class DynamicAugmentationConfig(BaseModel):
    """Dynamic augmentation configuration."""
    enabled: bool = True
    shift_range: float = 0.1
    rotation_range: float = 15
    blur_limit: int = 3
    blur_prob: float = 0.3
    sharpen_alpha: float = 0.2
    sharpen_lightness: float = 0.5
    sharpen_prob: float = 0.3
    intensity_offset: float = 0.1
    intensity_prob: float = 0.3
    contrast_scale: float = 0.1
    contrast_prob: float = 0.3
    gamma_range: List[float] = [0.8, 1.2]
    gamma_prob: float = 0.2
    elastic_alpha: float = 30
    elastic_sigma: float = 4
    elastic_prob: float = 0.2
    grid_distort_num_steps: int = 5
    grid_distort_range: float = 0.1
    grid_distort_prob: float = 0.2
    noise_scale: float = 0.05
    noise_prob: float = 0.2
    motion_blur_limit: int = 5
    motion_blur_prob: float = 0.1
    defocus_blur_radius: int = 3
    defocus_blur_prob: float = 0.1
    cutout_num_holes: int = 8
    cutout_max_h_size: int = 32
    cutout_min_h_size: int = 16
    cutout_prob: float = 0.1


class AugmentationConfig(BaseModel):
    """Augmentation configuration."""
    curriculum_epoch: int = 2
    static: StaticAugmentationConfig = Field(default_factory=StaticAugmentationConfig)
    dynamic: DynamicAugmentationConfig = Field(default_factory=DynamicAugmentationConfig)
    border_mode: str = "mirror"


class ExperimentConfig(BaseModel):
    """Experiment tracking configuration."""
    name: str = "default_experiment"
    log_dir: str = "outputs/experiments"
    checkpoint_dir: str = "outputs/checkpoints"
    prediction_dir: str = "outputs/predictions"
    visualization_dir: str = "outputs/visualizations"
    tensorboard: bool = True
    save_frequency: int = 10
    save_best_only: bool = True
    monitor_metric: str = "miou"


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""
    metrics: List[str] = ["miou", "dice", "f1", "precision", "recall", "accuracy"]
    visualize_predictions: bool = True
    num_visualization_samples: int = 10


class DeviceConfig(BaseModel):
    """Device configuration."""
    use_cuda: bool = False
    cuda_device: int = 0
    deterministic: bool = False
    seed: int = 42


class Config(BaseModel):
    """Main configuration class."""
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    classes: Dict[str, int] = {
        "background": 0,
        "die": 1,
        "scratches": 2,
        "contamination": 3,
        "other_defects": 4,
    }
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    
    class Config:
        arbitrary_types_allowed = True
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Load default config first
        default_config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
        if default_config_path.exists():
            with open(default_config_path, 'r') as f:
                default_dict = yaml.safe_load(f)
            # Merge configs (user config overrides defaults)
            config_dict = cls._merge_dicts(default_dict, config_dict)
        
        return cls(**config_dict)
    
    @staticmethod
    def _merge_dicts(default: Dict, override: Dict) -> Dict:
        """Recursively merge two dictionaries."""
        result = default.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._merge_dicts(result[key], value)
            else:
                result[key] = value
        return result
    
    def to_yaml(self, output_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.dict(), f, default_flow_style=False, sort_keys=False)
    
    def get_class_name(self, class_id: int) -> str:
        """Get class name from class ID."""
        for name, idx in self.classes.items():
            if idx == class_id:
                return name
        return f"class_{class_id}"
    
    def get_class_id(self, class_name: str) -> int:
        """Get class ID from class name."""
        if class_name not in self.classes:
            raise ValueError(f"Unknown class name: {class_name}")
        return self.classes[class_name]
    
    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        return len(self.classes)
    
    @property
    def class_names(self) -> List[str]:
        """Get list of class names sorted by ID."""
        return sorted(self.classes.keys(), key=lambda x: self.classes[x])
