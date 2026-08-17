"""SegFormer segmentation model wrapper."""
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn


class SegFormer(nn.Module):
    """SegFormer segmentation model wrapper.
    
    Supports SegFormer variants from Hugging Face Transformers:
    - segformer-b0 (smallest, fastest)
    - segformer-b1
    - segformer-b2
    - segformer-b3
    - segformer-b4
    - segformer-b5 (largest, most accurate)
    """
    
    def __init__(
        self,
        variant: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        num_classes: int = 5,
        pretrained: bool = True,
        input_size: Tuple[int, int] = (640, 640),
        ignore_index: int = 255,
    ):
        super().__init__()
        
        try:
            from transformers import SegformerForSemanticSegmentation
        except ImportError:
            raise ImportError(
                "Please install transformers: pip install transformers"
            )
        
        self.variant = variant
        self.num_classes = num_classes
        self.input_size = input_size
        self.ignore_index = ignore_index
        
        # Load SegFormer model
        if pretrained and "finetuned" in variant:
            # Load pretrained fine-tuned model and adapt head
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                variant,
                ignore_mismatched_sizes=True,
                num_labels=num_classes,
            )
        else:
            # Load base model or initialize from scratch
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                variant,
                num_labels=num_classes,
            )
        
        # Set input size expectations
        self.model.config.image_size = input_size
    
    def forward(
        self, 
        x: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            labels: Optional ground truth labels (B, H, W)
        
        Returns:
            Dictionary with 'logits' key containing segmentation logits
        """
        outputs = self.model(pixel_values=x, labels=labels)
        
        return {"logits": outputs.logits}
    
    def train_model(
        self,
        train_dataset: Any,
        val_dataset: Any,
        output_dir: str,
        num_train_epochs: int = 100,
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 50,
        device: str = "cpu",
        **kwargs: Any,
    ):
        """Train SegFormer using Hugging Face Trainer.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir: Output directory for checkpoints
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Training batch size
            per_device_eval_batch_size: Evaluation batch size
            learning_rate: Learning rate
            weight_decay: Weight decay
            save_steps: Save checkpoint every N steps
            eval_steps: Evaluate every N steps
            logging_steps: Log every N steps
            device: Device to use
            **kwargs: Additional training arguments
        
        Returns:
            Trainer and training results
        """
        from transformers import TrainingArguments, Trainer
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            save_steps=save_steps,
            eval_steps=eval_steps,
            logging_steps=logging_steps,
            evaluation_strategy="steps" if eval_steps > 0 else "no",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",  # Disable wandb/tensorboard if not needed
            **kwargs,
        )
        
        # Define compute_metrics function
        def compute_metrics(eval_pred):
            from .evaluation.metrics import compute_iou, compute_dice
            import numpy as np
            
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            
            iou = compute_iou(predictions, labels, self.num_classes)
            dice = compute_dice(predictions, labels, self.num_classes)
            
            return {
                "mIoU": float(np.nanmean(iou)),
                "mean_dice": float(np.nanmean(dice)),
            }
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        
        # Train
        trainer.train()
        
        return trainer
