"""Image dataset with indexed mask support for SegFormer."""
import os
import json
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from PIL import Image

from .base import BaseSegmentationDataset


class IndexedMaskDataset(BaseSegmentationDataset):
    """Dataset for SegFormer using indexed PNG masks.
    
    Expected structure:
    root_dir/
        images/
            img1.png
            img2.png
        masks/
            img1.png  # Grayscale with class indices
            img2.png
    """
    
    def __init__(
        self,
        root_dir: str,
        transforms: Optional[Any] = None,
        classes: Optional[Dict[str, int]] = None,
        input_size: Tuple[int, int] = (640, 640),
        image_subdir: str = "images",
        mask_subdir: str = "masks",
        ext: str = ".png",
    ):
        super().__init__(root_dir, transforms, classes, input_size)
        self.image_subdir = image_subdir
        self.mask_subdir = mask_subdir
        self.ext = ext
        
        self._load_paths()
    
    def _load_paths(self):
        """Load all image and mask paths."""
        images_dir = os.path.join(self.root_dir, self.image_subdir)
        masks_dir = os.path.join(self.root_dir, self.mask_subdir)
        
        if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
            raise FileNotFoundError(
                f"Expected directories not found: {images_dir}, {masks_dir}"
            )
        
        image_files = sorted([
            f for f in os.listdir(images_dir) 
            if f.endswith(self.ext) or f.endswith(".jpg") or f.endswith(".jpeg")
        ])
        
        for img_file in image_files:
            img_path = os.path.join(images_dir, img_file)
            mask_name = os.path.splitext(img_file)[0] + self.ext
            mask_path = os.path.join(masks_dir, mask_name)
            
            if os.path.exists(mask_path):
                self.image_paths.append(img_path)
                self.mask_paths.append(mask_path)
            else:
                print(f"Warning: Mask not found for {img_file}: {mask_path}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = self._load_image(img_path)
        mask = self._load_mask(mask_path)
        
        sample = {"image": image, "mask": mask}
        
        if self.transforms is not None:
            sample = self.transforms(**sample)
        
        image = sample["image"]
        mask = sample["mask"]
        
        # Convert to CHW format for PyTorch
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
        
        mask = mask.astype(np.int64)
        
        return {
            "image": image,
            "mask": mask,
            "image_path": img_path,
            "mask_path": mask_path,
        }
