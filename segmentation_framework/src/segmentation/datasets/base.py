"""Base dataset class for segmentation tasks."""
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class BaseSegmentationDataset(Dataset, ABC):
    """Abstract base class for segmentation datasets."""
    
    def __init__(
        self,
        root_dir: str,
        transforms: Optional[Any] = None,
        classes: Optional[Dict[str, int]] = None,
        input_size: Tuple[int, int] = (640, 640),
    ):
        self.root_dir = root_dir
        self.transforms = transforms
        self.classes = classes or {
            "background": 0,
            "die": 1,
            "scratches": 2,
            "contamination": 3,
            "other_defects": 4,
        }
        self.input_size = input_size
        self.image_paths = []
        self.mask_paths = []
        
    @abstractmethod
    def __len__(self) -> int:
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pass
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load image as RGB numpy array."""
        img = Image.open(path).convert("RGB")
        return np.array(img)
    
    def _load_mask(self, path: str) -> np.ndarray:
        """Load mask as grayscale numpy array."""
        mask = Image.open(path).convert("L")
        return np.array(mask)
