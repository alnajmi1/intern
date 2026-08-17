"""
Segmentation Framework for Industrial Inspection

A reusable, modular framework for training segmentation models
(YOLOv8, SegFormer, SAM) with curriculum learning augmentations.
"""

__version__ = "0.1.0"
__author__ = "Industrial AI Team"

from segmentation.config.config import Config
from segmentation.models.factory import create_model
from segmentation.models.yolo import YOLOv8Seg
from segmentation.models.segformer import SegFormer
from segmentation.datasets.loaders import create_data_loader, train_val_split
from segmentation.datasets.image_dataset import IndexedMaskDataset
from segmentation.datasets.coco_dataset import COCODataset
from segmentation.augmentation.pipeline import AugmentationPipeline
from segmentation.training.trainer import Trainer
from segmentation.evaluation.evaluator import Evaluator
from segmentation.evaluation.metrics import SegmentationMetrics
from segmentation.inference.predictor import Predictor

__all__ = [
    "Config",
    "create_model",
    "YOLOv8Seg",
    "SegFormer",
    "create_data_loader",
    "train_val_split",
    "IndexedMaskDataset",
    "COCODataset",
    "AugmentationPipeline",
    "Trainer",
    "Evaluator",
    "SegmentationMetrics",
    "Predictor",
]
