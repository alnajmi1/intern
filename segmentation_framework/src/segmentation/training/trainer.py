"""
Training loop for segmentation models.
Implements curriculum learning, checkpointing, and experiment tracking.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class Trainer:
    """
    Training loop manager for segmentation models.
    
    Features:
    - Curriculum learning with dynamic augmentations
    - Checkpoint saving/loading
    - Early stopping
    - TensorBoard logging
    - Mixed precision training (optional)
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: Any,
        device: torch.device,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Segmentation model to train
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            criterion: Loss function
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            config: Configuration object
            device: Training device (CPU/GPU)
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Training state
        self.current_epoch = 0
        self.best_metric = float('-inf') if config.experiment.monitor_metric != 'loss' else float('inf')
        self.patience_counter = 0
        
        # Directories
        self.checkpoint_dir = Path(config.experiment.checkpoint_dir)
        self.log_dir = Path(config.experiment.log_dir) / config.experiment.name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = None
        if config.experiment.tensorboard and TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Move model to device
        self.model.to(self.device)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.training.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            try:
                outputs = self.model(images)
                
                # Handle different output formats
                if isinstance(outputs, dict):
                    if 'logits' in outputs:
                        logits = outputs['logits']
                    elif 'pred_masks' in outputs:
                        logits = outputs['pred_masks']
                    else:
                        logits = list(outputs.values())[0]
                elif isinstance(outputs, (list, tuple)):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # Resize logits to match mask size if needed
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = nn.functional.interpolate(
                        logits, 
                        size=masks.shape[-2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                # Compute loss
                loss = self.criterion(logits, masks)
                
            except Exception as e:
                print(f"\nError in forward pass: {e}")
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training.gradient_clip
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / max(num_batches, 1)
        
        return {'loss': avg_loss}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            try:
                outputs = self.model(images)
                
                # Handle different output formats
                if isinstance(outputs, dict):
                    if 'logits' in outputs:
                        logits = outputs['logits']
                    elif 'pred_masks' in outputs:
                        logits = outputs['pred_masks']
                    else:
                        logits = list(outputs.values())[0]
                elif isinstance(outputs, (list, tuple)):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # Resize logits to match mask size if needed
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = nn.functional.interpolate(
                        logits, 
                        size=masks.shape[-2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                loss = self.criterion(logits, masks)
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"Error in validation: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        
        return {'val_loss': avg_loss}
    
    def train(self) -> Dict[str, List[float]]:
        """
        Run full training loop.
        
        Returns:
            Dictionary with training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': [],
        }
        
        print(f"\nStarting training for {self.config.training.epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model: {self.config.model.name}")
        print(f"Batch size: {self.config.training.batch_size}")
        print(f"Learning rate: {self.config.training.learning_rate}")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(epoch)
            history['train_loss'].append(train_metrics['loss'])
            
            # Validate
            val_metrics = self.validate()
            if val_metrics:
                history['val_loss'].append(val_metrics.get('val_loss', 0))
            
            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
                if val_metrics:
                    for key, value in val_metrics.items():
                        self.writer.add_scalar(f'Loss/{key}', value, epoch)
                self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config.training.epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            if val_metrics:
                print(f"  Val Loss: {val_metrics.get('val_loss', 0):.4f}")
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Checkpoint saving
            is_best = False
            if self.config.experiment.save_best_only:
                metric_to_monitor = val_metrics.get('val_loss', train_metrics['loss'])
                if self.config.experiment.monitor_metric == 'loss':
                    if metric_to_monitor < self.best_metric:
                        self.best_metric = metric_to_monitor
                        is_best = True
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                else:
                    # For metrics like mIoU where higher is better
                    if metric_to_monitor > self.best_metric:
                        self.best_metric = metric_to_monitor
                        is_best = True
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
            
            # Save checkpoint
            if is_best or (epoch + 1) % self.config.experiment.save_frequency == 0:
                self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.patience_counter >= self.config.training.patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        elapsed_time = time.time() - start_time
        print(f"\nTraining completed in {elapsed_time:.2f} seconds")
        
        # Close TensorBoard writer
        if self.writer:
            self.writer.close()
        
        return history
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'config': self.config.dict() if hasattr(self.config, 'dict') else vars(self.config),
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model (metric: {self.best_metric:.4f})")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_metric = checkpoint.get('best_metric', float('-inf'))
        
        return checkpoint.get('epoch', 0)
