"""
Augmentation pipeline for segmentation framework.
Implements curriculum learning with static and dynamic augmentations.
"""

import random
from typing import Any, Dict, Optional, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np


class AugmentationPipeline:
    """
    Augmentation pipeline with curriculum learning support.
    
    Applies static augmentations every epoch and dynamic augmentations
    every N epochs (configurable via curriculum_epoch).
    """
    
    def __init__(
        self,
        input_size: int = 640,
        num_classes: int = 5,
        curriculum_epoch: int = 2,
        static_config: Optional[Dict[str, Any]] = None,
        dynamic_config: Optional[Dict[str, Any]] = None,
        border_mode: str = "mirror",
        is_training: bool = True,
    ):
        """
        Initialize augmentation pipeline.
        
        Args:
            input_size: Target image size (will be resized to this)
            num_classes: Number of segmentation classes
            curriculum_epoch: Apply dynamic augmentations every N epochs
            static_config: Configuration for static augmentations
            dynamic_config: Configuration for dynamic augmentations
            border_mode: Border mode for geometric transforms ("mirror", "constant", etc.)
            is_training: Whether in training mode (applies augmentations)
        """
        self.input_size = input_size
        self.num_classes = num_classes
        self.curriculum_epoch = curriculum_epoch
        self.static_config = static_config or {}
        self.dynamic_config = dynamic_config or {}
        self.border_mode = self._get_border_mode(border_mode)
        self.is_training = is_training
        
        # Build base transform pipeline
        self.static_transform = self._build_static_transform()
        self.dynamic_transform = self._build_dynamic_transform()
    
    def _get_border_mode(self, border_mode: str) -> int:
        """Convert border mode string to OpenCV constant."""
        border_modes = {
            "mirror": cv2.BORDER_REFLECT_101,
            "reflect": cv2.BORDER_REFLECT,
            "replicate": cv2.BORDER_REPLICATE,
            "constant": cv2.BORDER_CONSTANT,
        }
        return border_modes.get(border_mode, cv2.BORDER_REFLECT_101)
    
    def _build_static_transform(self) -> A.Compose:
        """Build static augmentation pipeline."""
        transforms = []
        
        # Rotations (0, 90, 180, 270)
        if self.static_config.get("rotations"):
            rotations = self.static_config["rotations"]
            if len(rotations) > 1:
                transforms.append(
                    A.RandomRotate90(p=self.static_config.get("prob_rotation", 0.5))
                )
        
        # Flips
        if self.static_config.get("flips"):
            flips = self.static_config["flips"]
            h_flip = "x" in flips or "xy" in flips
            v_flip = "y" in flips or "xy" in flips
            
            if h_flip and v_flip:
                transforms.append(
                    A.Flip(p=self.static_config.get("prob_flip", 0.5))
                )
            elif h_flip:
                transforms.append(
                    A.HorizontalFlip(p=self.static_config.get("prob_flip", 0.5))
                )
            elif v_flip:
                transforms.append(
                    A.VerticalFlip(p=self.static_config.get("prob_flip", 0.5))
                )
        
        # Random crop
        if self.static_config.get("random_crop", False):
            crop_ratio = self.static_config.get("crop_size_ratio", 0.9)
            transforms.append(
                A.RandomCrop(
                    height=int(self.input_size * crop_ratio),
                    width=int(self.input_size * crop_ratio),
                    p=self.static_config.get("prob_crop", 0.3),
                )
            )
        
        # Resize to target size
        transforms.append(A.Resize(height=self.input_size, width=self.input_size))
        
        # Normalize and convert to tensor
        transforms.extend([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        
        return A.Compose(transforms, border_mode=self.border_mode)
    
    def _build_dynamic_transform(self) -> A.Compose:
        """Build dynamic augmentation pipeline (curriculum learning)."""
        if not self.dynamic_config.get("enabled", True):
            return A.Compose([])
        
        transforms = []
        
        # Small XY shift
        shift_range = self.dynamic_config.get("shift_range", 0.1)
        if shift_range > 0:
            transforms.append(
                A.ShiftScaleRotate(
                    shift_limit=(shift_range, shift_range),
                    scale_limit=0,
                    rotate_limit=0,
                    p=0.5,
                    border_mode=self.border_mode,
                )
            )
        
        # Small rotation
        rotation_range = self.dynamic_config.get("rotation_range", 15)
        if rotation_range > 0:
            transforms.append(
                A.ShiftScaleRotate(
                    shift_limit=0,
                    scale_limit=0,
                    rotate_limit=(-rotation_range, rotation_range),
                    p=0.5,
                    border_mode=self.border_mode,
                )
            )
        
        # Gaussian blur
        blur_limit = self.dynamic_config.get("blur_limit", 3)
        blur_prob = self.dynamic_config.get("blur_prob", 0.3)
        if blur_limit > 0:
            transforms.append(A.GaussianBlur(blur_limit=(3, blur_limit), p=blur_prob))
        
        # Sharpening (unsharp mask)
        sharpen_alpha = self.dynamic_config.get("sharpen_alpha", 0.2)
        sharpen_lightness = self.dynamic_config.get("sharpen_lightness", 0.5)
        sharpen_prob = self.dynamic_config.get("sharpen_prob", 0.3)
        if sharpen_alpha > 0:
            transforms.append(
                A.Sharpen(alpha=(0.2, sharpen_alpha), lightness=(0.5, sharpen_lightness), p=sharpen_prob)
            )
        
        # Intensity offsets
        intensity_offset = self.dynamic_config.get("intensity_offset", 0.1)
        intensity_prob = self.dynamic_config.get("intensity_prob", 0.3)
        if intensity_offset > 0:
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=(-intensity_offset, intensity_offset),
                    contrast_limit=0,
                    p=intensity_prob,
                )
            )
        
        # Contrast stretch (scaling)
        contrast_scale = self.dynamic_config.get("contrast_scale", 0.1)
        contrast_prob = self.dynamic_config.get("contrast_prob", 0.3)
        if contrast_scale > 0:
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=0,
                    contrast_limit=(-contrast_scale, contrast_scale),
                    p=contrast_prob,
                )
            )
        
        # Gamma correction
        gamma_range = self.dynamic_config.get("gamma_range", [0.8, 1.2])
        gamma_prob = self.dynamic_config.get("gamma_prob", 0.2)
        if gamma_range[1] > gamma_range[0]:
            transforms.append(A.RandomGamma(gamma_limit=gamma_range, p=gamma_prob))
        
        # Elastic transform
        elastic_alpha = self.dynamic_config.get("elastic_alpha", 30)
        elastic_sigma = self.dynamic_config.get("elastic_sigma", 4)
        elastic_prob = self.dynamic_config.get("elastic_prob", 0.2)
        if elastic_alpha > 0:
            transforms.append(
                A.ElasticTransform(
                    alpha=elastic_alpha,
                    sigma=elastic_sigma,
                    alpha_affine=0,
                    p=elastic_prob,
                    border_mode=self.border_mode,
                )
            )
        
        # Grid distortion
        grid_steps = self.dynamic_config.get("grid_distort_num_steps", 5)
        grid_range = self.dynamic_config.get("grid_distort_range", 0.1)
        grid_prob = self.dynamic_config.get("grid_distort_prob", 0.2)
        if grid_steps > 0:
            transforms.append(
                A.GridDistortion(num_steps=grid_steps, distort_limit=grid_range, p=grid_prob)
            )
        
        # Noise
        noise_scale = self.dynamic_config.get("noise_scale", 0.05)
        noise_prob = self.dynamic_config.get("noise_prob", 0.2)
        if noise_scale > 0:
            transforms.append(A.IAASuperpixels(p=noise_prob, n_segments=int(100 / noise_scale)))
            transforms.append(A.GaussNoise(var_limit=(noise_scale * 255)**2, p=noise_prob))
        
        # Motion blur
        motion_limit = self.dynamic_config.get("motion_blur_limit", 5)
        motion_prob = self.dynamic_config.get("motion_blur_prob", 0.1)
        if motion_limit > 0:
            transforms.append(A.MotionBlur(blur_limit=motion_limit, p=motion_prob))
        
        # Defocus blur
        defocus_radius = self.dynamic_config.get("defocus_blur_radius", 3)
        defocus_prob = self.dynamic_config.get("defocus_blur_prob", 0.1)
        if defocus_radius > 0:
            transforms.append(A.Defocus(radius=(1, defocus_radius), p=defocus_prob))
        
        # Cutout / Coarse dropout
        cutout_holes = self.dynamic_config.get("cutout_num_holes", 8)
        cutout_max_size = self.dynamic_config.get("cutout_max_h_size", 32)
        cutout_min_size = self.dynamic_config.get("cutout_min_h_size", 16)
        cutout_prob = self.dynamic_config.get("cutout_prob", 0.1)
        if cutout_holes > 0:
            transforms.append(
                A.CoarseDropout(
                    max_holes=cutout_holes,
                    max_height=cutout_max_size,
                    max_width=cutout_max_size,
                    min_holes=cutout_holes // 2,
                    min_height=cutout_min_size,
                    min_width=cutout_min_size,
                    p=cutout_prob,
                )
            )
        
        return A.Compose(transforms, border_mode=self.border_mode)
    
    def should_apply_dynamic(self, epoch: int) -> bool:
        """Check if dynamic augmentations should be applied for given epoch."""
        if not self.dynamic_config.get("enabled", True):
            return False
        return epoch % self.curriculum_epoch == 0
    
    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        epoch: int = 0,
    ) -> Dict[str, np.ndarray]:
        """
        Apply augmentations to image and mask.
        
        Args:
            image: Input image (H, W, C) in RGB format
            mask: Segmentation mask (H, W) with class indices
            epoch: Current training epoch (for curriculum learning)
        
        Returns:
            Dictionary with 'image' and optionally 'mask' keys
        """
        if not self.is_training:
            # Validation/test mode - only resize and normalize
            result = self.static_transform(image=image, mask=mask)
            return {"image": result["image"], "mask": result["mask"]}
        
        # Apply static augmentations first
        result = self.static_transform(image=image, mask=mask)
        
        # Apply dynamic augmentations if it's a curriculum epoch
        if self.should_apply_dynamic(epoch):
            # Dynamic transforms are applied before final resize/normalize
            # So we need to temporarily reverse the normalization
            temp_image = result["image"].numpy().transpose(1, 2, 0)
            temp_image = (temp_image * np.array([0.229, 0.224, 0.225]) + 
                         np.array([0.485, 0.456, 0.406]))
            temp_image = np.clip(temp_image, 0, 1)
            
            temp_mask = result["mask"].numpy()
            
            dynamic_result = self.dynamic_transform(image=temp_image, mask=temp_mask)
            
            # Re-apply normalization and convert to tensor
            result = self.static_transform(
                image=dynamic_result["image"],
                mask=dynamic_result["mask"],
            )
        
        return {"image": result["image"], "mask": result["mask"]}
    
    def set_training(self, mode: bool) -> None:
        """Set training mode."""
        self.is_training = mode
