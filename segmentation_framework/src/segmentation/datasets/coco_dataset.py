"""COCO format dataset loader for YOLOv8."""
import os
import json
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from PIL import Image
import cv2

from .base import BaseSegmentationDataset


class COCODataset(BaseSegmentationDataset):
    """Dataset loader for COCO JSON format (CVAT export).
    
    Expected structure:
    root_dir/
        images/
            img1.png
            img2.png
        annotations.json  # COCO format with segmentation polygons/masks
    """
    
    def __init__(
        self,
        root_dir: str,
        transforms: Optional[Any] = None,
        classes: Optional[Dict[str, int]] = None,
        input_size: Tuple[int, int] = (640, 640),
        image_subdir: str = "images",
        annotation_file: str = "annotations.json",
    ):
        super().__init__(root_dir, transforms, classes, input_size)
        self.image_subdir = image_subdir
        self.annotation_file = annotation_file
        
        self.coco_data = None
        self.img_id_to_path = {}
        self.img_anns = {}
        
        self._load_annotations()
    
    def _load_annotations(self):
        """Load COCO JSON annotations."""
        ann_path = os.path.join(self.root_dir, self.annotation_file)
        
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")
        
        with open(ann_path, "r") as f:
            self.coco_data = json.load(f)
        
        images_dir = os.path.join(self.root_dir, self.image_subdir)
        
        # Build image ID to path mapping
        for img_info in self.coco_data.get("images", []):
            img_id = img_info["id"]
            img_file = img_info["file_name"]
            img_path = os.path.join(images_dir, img_file)
            
            if os.path.exists(img_path):
                self.img_id_to_path[img_id] = img_path
            else:
                print(f"Warning: Image not found: {img_path}")
        
        # Group annotations by image ID
        for ann in self.coco_data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in self.img_anns:
                self.img_anns[img_id] = []
            self.img_anns[img_id].append(ann)
        
        self.image_paths = list(self.img_id_to_path.values())
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def _polygons_to_mask(self, polygons: List, height: int, width: int) -> np.ndarray:
        """Convert COCO polygons to binary mask."""
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for poly in polygons:
            if isinstance(poly, dict) and "counts" in poly:
                # RLE format - skip for now, handle if needed
                continue
            
            # Polygon format: [x1, y1, x2, y2, ...]
            pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 1)
        
        return mask
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.image_paths[idx]
        
        # Find image ID from path
        img_id = None
        for iid, ipath in self.img_id_to_path.items():
            if ipath == img_path:
                img_id = iid
                break
        
        if img_id is None:
            raise ValueError(f"Image ID not found for {img_path}")
        
        image = self._load_image(img_path)
        height, width = image.shape[:2]
        
        # Create mask from annotations
        mask = np.zeros((height, width), dtype=np.uint8)
        
        annotations = self.img_anns.get(img_id, [])
        
        # Build category ID mapping (COCO cat_id -> our class index)
        cat_id_map = {}
        for cat in self.coco_data.get("categories", []):
            cat_name = cat["name"]
            if cat_name in self.classes:
                cat_id_map[cat["id"]] = self.classes[cat_name]
        
        for ann in annotations:
            cat_id = ann["category_id"]
            class_idx = cat_id_map.get(cat_id, 0)
            
            if class_idx == 0:  # background, skip or handle separately
                continue
            
            seg = ann.get("segmentation", [])
            if isinstance(seg, list):
                # Polygons
                poly_mask = self._polygons_to_mask(seg, height, width)
                mask[poly_mask > 0] = class_idx
            elif isinstance(seg, dict) and "counts" in seg:
                # RLE format - implement if needed
                pass
        
        sample = {"image": image, "mask": mask}
        
        if self.transforms is not None:
            sample = self.transforms(**sample)
        
        image = sample["image"]
        mask = sample["mask"]
        
        # Convert to CHW format
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
        
        mask = mask.astype(np.int64)
        
        return {
            "image": image,
            "mask": mask,
            "image_path": img_path,
            "num_classes": len(self.classes),
        }
