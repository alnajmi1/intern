"""Inference predictor for segmentation models."""
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import torch
from PIL import Image
import cv2


class Predictor:
    """Inference handler for segmentation models.
    
    Provides easy-to-use prediction interface for trained models with
    preprocessing and postprocessing capabilities.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        input_size: Tuple[int, int] = (640, 640),
        num_classes: int = 5,
    ):
        self.model = model
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        
        self.model.eval()
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input.
        
        Args:
            image: Input image (H, W, 3) in uint8
        
        Returns:
            Preprocessed tensor (1, C, H, W)
        """
        # Resize to input size
        img_resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert to float and normalize
        img_float = img_resized.astype(np.float32) / 255.0
        
        # Convert to CHW format
        if img_float.ndim == 2:
            img_float = np.stack([img_float] * 3, axis=-1)
        img_chw = np.transpose(img_float, (2, 0, 1))
        
        # Add batch dimension and convert to tensor
        img_tensor = torch.from_numpy(img_chw).unsqueeze(0).to(self.device)
        
        return img_tensor
    
    def postprocess(self, output: torch.Tensor) -> np.ndarray:
        """Convert model output to segmentation mask.
        
        Args:
            output: Model output logits/tensor
        
        Returns:
            Segmentation mask (H, W) with class indices
        """
        # Handle different output formats
        if isinstance(output, dict):
            if "masks" in output:
                logits = output["masks"]
            elif "logits" in output:
                logits = output["logits"]
            else:
                logits = list(output.values())[0]
        else:
            logits = output
        
        # Get class predictions
        if logits.ndim == 4:
            # (B, C, H, W) -> (B, H, W)
            pred = logits.argmax(dim=1).cpu().numpy()
        else:
            pred = logits.cpu().numpy()
        
        # Remove batch dimension
        if pred.ndim == 3:
            pred = pred[0]
        
        return pred
    
    @torch.no_grad()
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run inference on a single image.
        
        Args:
            image: Input image (H, W, 3) in uint8
        
        Returns:
            Segmentation mask (H, W) with class indices
        """
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Postprocess
        mask = self.postprocess(output)
        
        # Resize back to original size if needed
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(
                mask, 
                (image.shape[1], image.shape[0]), 
                interpolation=cv2.INTER_NEAREST
            )
        
        return mask
    
    @torch.no_grad()
    def predict_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Run inference on a batch of images.
        
        Args:
            images: List of input images
        
        Returns:
            List of segmentation masks
        """
        # Preprocess all images
        input_tensors = [self.preprocess(img) for img in images]
        batch_tensor = torch.cat(input_tensors, dim=0)
        
        # Forward pass
        outputs = self.model(batch_tensor)
        
        # Postprocess all predictions
        masks = [self.postprocess(outputs[i:i+1]) for i in range(len(images))]
        
        # Resize back to original sizes
        resized_masks = []
        for mask, img in zip(masks, images):
            if mask.shape != img.shape[:2]:
                mask = cv2.resize(
                    mask,
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            resized_masks.append(mask)
        
        return resized_masks
    
    @torch.no_grad()
    def predict_with_confidence(
        self, 
        image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference and return both mask and confidence scores.
        
        Args:
            image: Input image
        
        Returns:
            Tuple of (mask, confidence_map)
        """
        input_tensor = self.preprocess(image)
        output = self.model(input_tensor)
        
        # Handle different output formats
        if isinstance(output, dict):
            if "masks" in output:
                logits = output["masks"]
            elif "logits" in output:
                logits = output["logits"]
            else:
                logits = list(output.values())[0]
        else:
            logits = output
        
        # Get probabilities via softmax
        probs = torch.softmax(logits, dim=1)
        
        # Get max probability (confidence) and argmax (prediction)
        confidence, pred_class = torch.max(probs, dim=1)
        
        mask = pred_class.cpu().numpy()[0]
        confidence_map = confidence.cpu().numpy()[0]
        
        # Resize if needed
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            confidence_map = cv2.resize(
                confidence_map,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        
        return mask, confidence_map
