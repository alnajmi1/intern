"""
Segmentation Framework for Industrial Inspection
"""

__version__ = "0.1.0"
__author__ = "Industrial AI Team"

from segmentation.config.config import Config
from segmentation.models.factory import create_model
from segmentation.datasets.loaders import get_dataloader
from segmentation.augmentation.pipeline import AugmentationPipeline
from segmentation.training.trainer import Trainer
from segmentation.evaluation.evaluator import Evaluator
from segmentation.inference.predictor import Predictor

__all__ = [
    "Config",
    "create_model",
    "get_dataloader",
    "AugmentationPipeline",
    "Trainer",
    "Evaluator",
    "Predictor",
]
