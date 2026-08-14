"""
SegFormer Tutorial for Beginners
=================================
This script demonstrates how to use SegFormer for semantic segmentation
using dummy data. No prior experience required!

SegFormer is a transformer-based model for semantic segmentation that
divides images into patches and processes them efficiently.

What this script does:
1. Creates dummy images and segmentation masks
2. Loads a pre-trained SegFormer model
3. Trains the model on our dummy data (just 1 epoch for demonstration)
4. Makes a prediction to show it works
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from PIL import Image
import numpy as np
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Limit CPU threads to reduce memory usage
torch.set_num_threads(1)

print("=" * 60)
print("SegFormer Tutorial - Starting Up!")
print("=" * 60)

# =============================================================================
# STEP 1: Create Dummy Dataset
# =============================================================================
print("\n[Step 1] Creating dummy dataset...")

class DummySegmentationDataset(Dataset):
    """
    A simple dataset that generates random images and masks.
    
    In real applications, you would load actual images from disk.
    Here we create synthetic data to demonstrate the workflow.
    """
    
    def __init__(self, num_samples=50, image_size=(512, 512), num_classes=5):
        """
        Args:
            num_samples: How many dummy images to create
            image_size: Size of images (height, width)
            num_classes: Number of segmentation classes
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        
        print(f"  → Creating {num_samples} dummy images")
        print(f"  → Image size: {image_size}")
        print(f"  → Number of classes: {num_classes}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Create a random RGB image
        # Values between 0-255, converted to float for PIL
        image_array = np.random.randint(0, 256, 
                                        (*self.image_size, 3), 
                                        dtype=np.uint8)
        image = Image.fromarray(image_array, mode='RGB')
        
        # Create a random segmentation mask
        # Each pixel gets a random class label (0 to num_classes-1)
        mask_array = np.random.randint(0, self.num_classes, 
                                       self.image_size, 
                                       dtype=np.int64)
        mask = Image.fromarray(mask_array, mode='L')  # 'L' = grayscale/single channel
        
        return {
            'image': image,
            'mask': mask
        }

# Create training and validation datasets with smaller images to save memory
train_dataset = DummySegmentationDataset(num_samples=20, image_size=(128, 128), num_classes=5)
val_dataset = DummySegmentationDataset(num_samples=5, image_size=(128, 128), num_classes=5)

print("  ✓ Dummy datasets created!")

# =============================================================================
# STEP 2: Load Pre-trained SegFormer Model and Processor
# =============================================================================
print("\n[Step 2] Loading SegFormer model and image processor...")

# Choose a SegFormer variant (b0 is the smallest/fastest)
model_name = "nvidia/segformer-b0-finetuned-ade-512-512"

print(f"  → Loading model: {model_name}")
print("  → This may take a moment if downloading for the first time...")

# Image processor handles resizing, normalization, etc.
# We override the default size to use smaller images for memory efficiency
image_processor = SegformerImageProcessor.from_pretrained(
    model_name, 
    size={"height": 128, "width": 128},
    crop_size={"height": 128, "width": 128}
)

# The actual segmentation model
model = SegformerForSemanticSegmentation.from_pretrained(model_name)

# Update the model's classifier to match our number of classes (5 in our case)
# The pre-trained model has 150 classes (ADE20K dataset), we need only 5
model.config.num_labels = 5
model.classifier = torch.nn.Conv2d(
    model.config.decoder_hidden_size,
    5,  # Our number of classes
    kernel_size=1
)

print(f"  ✓ Model loaded and adapted for {5} classes!")
print(f"  → Model architecture: SegFormer B0")
print(f"  → Image size: 128x128 (optimized for low-memory systems)")

# =============================================================================
# STEP 3: Prepare Data Loader with Preprocessing
# =============================================================================
print("\n[Step 3] Setting up data preprocessing and loaders...")

def collate_fn(batch):
    """
    Custom function to process a batch of images and masks.
    This is where we apply the image processor to prepare data for the model.
    """
    images = [item['image'] for item in batch]
    masks = [item['mask'] for item in batch]
    
    # Process images: resize, normalize, convert to tensors
    # Use 'segmentation_maps' parameter for masks (not 'masks')
    encoding = image_processor(images=images, segmentation_maps=masks, return_tensors="pt")
    
    return encoding

# Create data loaders
batch_size = 4

train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    collate_fn=collate_fn,
    num_workers=0  # Keep at 0 for simplicity on Windows/single GPU
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    collate_fn=collate_fn,
    num_workers=0
)

print(f"  → Batch size: {batch_size}")
print(f"  → Training batches: {len(train_loader)}")
print(f"  → Validation batches: {len(val_loader)}")
print("  ✓ Data loaders ready!")

# =============================================================================
# STEP 4: Training Setup
# =============================================================================
print("\n[Step 4] Configuring training...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  → Using device: {device}")

model.to(device)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

print("  → Loss function: Cross Entropy Loss")
print("  → Optimizer: AdamW")
print("  → Learning rate: 0.00005")

# =============================================================================
# STEP 5: Training Loop (1 Epoch for Demonstration)
# =============================================================================
print("\n[Step 5] Starting training (1 epoch for demo)...")
print("-" * 60)

model.train()  # Set model to training mode

total_loss = 0
num_batches = 0

for batch_idx, batch in enumerate(train_loader):
    # Move data to GPU/CPU
    pixel_values = batch['pixel_values'].to(device)
    labels = batch['labels'].to(device)
    
    # Forward pass
    outputs = model(pixel_values=pixel_values, labels=labels)
    loss = outputs.loss
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Track metrics
    total_loss += loss.item()
    num_batches += 1
    
    # Print progress every 2 batches
    if (batch_idx + 1) % 2 == 0:
        print(f"  Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

avg_train_loss = total_loss / num_batches
print("-" * 60)
print(f"  ✓ Training complete! Average loss: {avg_train_loss:.4f}")

# =============================================================================
# STEP 6: Validation/Evaluation
# =============================================================================
print("\n[Step 6] Running validation...")

model.eval()  # Set model to evaluation mode

val_loss = 0
val_batches = 0

with torch.no_grad():  # No gradient computation needed
    for batch in val_loader:
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        
        val_loss += loss.item()
        val_batches += 1

avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
print(f"  ✓ Validation complete! Average loss: {avg_val_loss:.4f}")

# =============================================================================
# STEP 7: Make a Prediction (Inference)
# =============================================================================
print("\n[Step 7] Making a prediction on a dummy image...")

# Create a single dummy image for inference (use same size as training)
dummy_image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
dummy_image_pil = Image.fromarray(dummy_image, mode='RGB')

# Preprocess the image
inputs = image_processor(images=dummy_image_pil, return_tensors="pt").to(device)

# Run inference
with torch.no_grad():
    outputs = model(**inputs)

# Get predictions
logits = outputs.logits  # Raw predictions
predictions = logits.argmax(dim=1)  # Get class with highest probability

print(f"  → Input image shape: {dummy_image_pil.size}")
print(f"  → Output logits shape: {logits.shape}")
print(f"  → Prediction shape: {predictions.shape}")
print(f"  → Unique predicted classes: {torch.unique(predictions).tolist()}")
print("  ✓ Inference successful!")

# =============================================================================
# STEP 8: Save the Model (Optional)
# =============================================================================
print("\n[Step 8] Saving the trained model...")

save_path = "./segformer_dummy_model"
model.save_pretrained(save_path)
image_processor.save_pretrained(save_path)

print(f"  ✓ Model saved to: {save_path}")
print("  → You can reload it later with:")
print(f"     model = SegformerForSemanticSegmentation.from_pretrained('{save_path}')")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("TUTORIAL COMPLETE!")
print("=" * 60)
print("""
What you learned:
1. ✓ How to create a custom dataset for segmentation
2. ✓ How to load SegFormer from Hugging Face Transformers
3. ✓ How to adapt the model for your number of classes
4. ✓ How to preprocess images with SegformerImageProcessor
5. ✓ How to train the model (even on dummy data!)
6. ✓ How to make predictions

Next steps for real projects:
• Replace dummy data with actual images and masks
• Increase training epochs (we only did 1 for demo)
• Add proper evaluation metrics (IoU, Pixel Accuracy)
• Use data augmentation for better generalization
• Consider using a larger SegFormer variant (b1, b2, etc.)

Resources:
• Hugging Face Docs: https://huggingface.co/docs/transformers/model_doc/segformer
• Original Paper: https://arxiv.org/abs/2105.15203
• NVIDIA SegFormer: https://github.com/NVlabs/SegFormer
""")
print("=" * 60)
