import os
import csv
import torch
import torch.nn as nn
from ultralytics import YOLO
import gc
import time
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pytorch_metric_learning.losses import NTXentLoss

# ====================================
# CONFIGURATION
# ====================================

# Model variants to test
YOLO_MODELS = [
    #"yolov8n.yaml", # Nano
    #"yolov8s.yaml", # Small
    "yolov8m.yaml", # Medium
    "yolov8l.yaml", # Large
    "yolov8x.yaml"  # Extra Large
]

# Image sizes to test (multiples of 32 for YOLO)
IMAGE_SIZES = [
    224,
    320,
    416,
    512,
    640,
    #768,
    #896,
    #1024
]

# Initial batch sizes to test
INITIAL_BATCH_SIZES = {
    #"yolov8n.yaml": 128,
    #"yolov8s.yaml": 64,
    "yolov8m.yaml": 32,
    "yolov8l.yaml": 16,
    "yolov8x.yaml": 8
}

# Number of backbone layers to include
BACKBONE_LAYERS = 11

# Directory containing images for testing
DEFAULT_IMG_DIR = "/home/recherche-a/OneDrive_recherche_a/Linux_onedrive/Datasets/steel-common-aug/images/train"

# Output file
OUTPUT_CSV = "batch_size_results.csv"

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training Epochs
TRAINING_EPOCHS = 2

# ====================================
# DATASET CLASS
# ====================================

class SimpleImageDataset(Dataset):
    """Simple dataset for loading a few images"""
    
    def __init__(self, img_dir, transform=None, max_images=100):
        self.img_paths = list(Path(img_dir).glob("*.jpg"))[:max_images]
        self.transform = transform
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
            
        return img

# ====================================
# UTILITY FUNCTIONS
# ====================================

def reset_gpu_memory():
    """Clear GPU cache and run garbage collection"""
    torch.cuda.empty_cache()
    gc.collect()

# ====================================
# MODEL AND TESTING FUNCTIONS
# ====================================

class SimYOLOv8ForTesting(nn.Module):
    """Simplified YOLOv8 for SSL training"""
    
    def __init__(self, backbone, img_size, model_yaml):
        super().__init__()
        self.backbone = backbone
        
        # Determine input features based on the model YAML name
        model_type = Path(model_yaml).stem
        print(f"Model type from YAML: {model_type}")
        
        if 'yolov8x' in model_type:
            in_features = 640
        elif 'yolov8l' in model_type:
            in_features = 512
        elif 'yolov8m' in model_type:
            in_features = 576
        elif 'yolov8s' in model_type:
            in_features = 512
        elif 'yolov8n' in model_type:
            in_features = 256
        else:
            in_features = 512
            print(f"Warning: Unknown model type '{model_type}'. Using default in_features={in_features}")
        
        print(f"Using in_features={in_features} for model {model_type}")
        
        # Projection head
        out_features = 256
        hidden_dim = out_features * 2
        
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_features, bias=True),
            nn.BatchNorm1d(out_features, affine=False)
        )
        
        # Create transform
        self.transform = v2.Compose([
            v2.RandomResizedCrop(size=img_size, scale=(0.7, 1.0)),
            v2.ColorJitter(brightness=0.1, contrast=0.2),
            v2.RandomRotation(degrees=30),
            v2.ConvertImageDtype(),
        ])

    def forward(self, x):
        # Get 2 augmentations of the batch
        augm_1 = self.transform(x)
        augm_2 = self.transform(x)

        # Process views through backbone and projection head
        h_1 = self.backbone(augm_1)
        h_2 = self.backbone(augm_2)
        
        compact_h_1 = self.mlp(h_1)
        compact_h_2 = self.mlp(h_2)
        
        return compact_h_1, compact_h_2

def create_yolo_backbone(model_yaml):
    """Extract backbone layers from YOLOv8 model"""
    model = YOLO(model_yaml)
    model_children_list = list(model.model.children())
    backbone = model_children_list[0][:BACKBONE_LAYERS]
    return backbone

def test_configuration(model_yaml, img_size, batch_size, img_dir):
    """Test if a configuration works by training for TRAINING_EPOCHS"""
    try:
        reset_gpu_memory()
        
        # Create data transform
        transform = v2.Compose([
            v2.Resize(size=(img_size, img_size)),
            v2.ToImageTensor(),
            v2.ConvertImageDtype(dtype=torch.float32),
        ])
        
        # Load data
        dataset = SimpleImageDataset(img_dir, transform=transform, max_images=200)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)
        
        # Create model
        backbone = create_yolo_backbone(model_yaml)
        model = SimYOLOv8ForTesting(backbone, img_size, model_yaml).to(DEVICE)
        model.train()
        
        # Create loss function and optimizer
        loss_fn = NTXentLoss(temperature=0.25)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Actual training loop
        for epoch in range(TRAINING_EPOCHS):
            print(f"Training epoch {epoch+1}/{TRAINING_EPOCHS} for {model_yaml}, size {img_size}, batch {batch_size}")
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Batch progress")):
                data = batch.to(DEVICE)
                optimizer.zero_grad()
                
                # Forward pass
                emb1, emb2 = model(data)
                
                # Prepare embeddings for loss
                embeddings = torch.cat([emb1, emb2], dim=0)
                
                # Use the actual batch size from emb1 instead of the configured batch_size
                # This is important for the last batch which might be smaller
                actual_batch_size = emb1.size(0)
                indices = torch.arange(actual_batch_size, device=DEVICE)
                labels = torch.cat([indices, indices])
                
                # Calculate loss
                loss = loss_fn(embeddings, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Show batch progress periodically
                if batch_idx % 5 == 0:
                    print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
                
            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1} complete. Average loss: {avg_epoch_loss:.4f}")
        
        # Cleanup
        del model, backbone, optimizer, data, embeddings, loss
        reset_gpu_memory()
        
        print(f"✅ SUCCESS: Model: {model_yaml}, Size: {img_size}, Batch: {batch_size}")
        return {
            'model': Path(model_yaml).stem,
            'image_size': img_size,
            'batch_size': batch_size,
            'status': 'success'
        }
        
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f"❌ OUT OF MEMORY: Model: {model_yaml}, Size: {img_size}, Batch: {batch_size}")
            # Cleanup after OOM
            reset_gpu_memory()
            return {
                'model': Path(model_yaml).stem,
                'image_size': img_size,
                'batch_size': batch_size,
                'status': 'OOM'
            }
        else:
            # Other runtime errors
            print(f"⚠️ ERROR: {e}")
            reset_gpu_memory()
            return {
                'model': Path(model_yaml).stem,
                'image_size': img_size,
                'batch_size': batch_size,
                'status': f'error: {str(e)[:100]}...'
            }

def find_max_batch_size(model_yaml, img_size, img_dir):
    """Binary search to find maximum batch size that fits in memory"""
    
    # Initial batch size guess based on model
    initial_batch = INITIAL_BATCH_SIZES.get(model_yaml, 8)
    min_batch = 1
    max_batch = initial_batch
    
    print(f"\nFinding max batch size for {model_yaml} at {img_size}x{img_size}")
    
    # Test initial batch size first
    result = test_configuration(model_yaml, img_size, initial_batch, img_dir)
    if result['status'] != 'success':
        # If initial size fails, fall back to binary search starting from half
        max_batch = max(2, initial_batch // 2)
    else:
        # If initial size works, incrementally increase until we find the limit
        while True:
            next_batch = max_batch + max(1, max_batch // 4)  # Increase by 25% each time
            result = test_configuration(model_yaml, img_size, next_batch, img_dir)
            if result['status'] != 'success':
                break
                
            max_batch = next_batch
            if max_batch >= 512:  # Arbitrary upper limit
                break
    
    # Binary search between min_batch and max_batch
    while max_batch - min_batch > 1:
        mid_batch = (min_batch + max_batch) // 2
        result = test_configuration(model_yaml, img_size, mid_batch, img_dir)
        
        if result['status'] == 'success':
            min_batch = mid_batch  # This batch size works
        else:
            max_batch = mid_batch  # This batch size causes OOM
    
    # Final verification with best found batch size
    final_result = test_configuration(model_yaml, img_size, min_batch, img_dir)
    
    # If final result is successful, we use that as our answer
    if final_result['status'] == 'success':
        return final_result
    
    # If stable batch size also fails, try one more time with the minimum
    if min_batch > 1:
        return test_configuration(model_yaml, img_size, 1, img_dir)
    
    # Nothing worked, return the original final result
    return final_result

# ====================================
# MAIN FUNCTION
# ====================================

def main(args):
    print(f"Testing maximum batch sizes for YOLO models by training for {TRAINING_EPOCHS} epochs")
    print(f"Output will be saved to: {args.output}")
    print(f"Using images from: {args.img_dir}")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This script requires a GPU.")
        return
    
    # Display GPU info
    gpu_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
    print(f"GPU: {gpu_name}, Total Memory: {total_mem:.2f} MB")
    
    results = []
    
    # For each model and image size, find the max batch size
    for model_yaml in tqdm(YOLO_MODELS, desc="Testing models"):
        for img_size in tqdm(IMAGE_SIZES, desc=f"Testing {model_yaml} image sizes", leave=False):
            result = find_max_batch_size(model_yaml, img_size, args.img_dir)
            results.append(result)
    
    # Save results to CSV
    with open(args.output, 'w', newline='') as csvfile:
        fieldnames = ['model', 'image_size', 'batch_size', 'status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"\nResults saved to {args.output}")
    
    # Print a summary table
    print("\nSummary of maximum batch sizes:")
    print("-" * 70)
    print(f"{'Model':<10} | {'Image Size':<10} | {'Maximum Batch Size':<18} | {'Status':<10}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['model']:<10} | {result['image_size']:<10} | {result['batch_size']:<18} | {result['status']:<10}")
    
    print("-" * 70)
    print("Note: These batch sizes have been verified by actual training for 2 epochs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find maximum batch sizes by training models for 2 epochs")
    parser.add_argument("--img_dir", type=str, default=DEFAULT_IMG_DIR, help="Directory containing images for testing")
    parser.add_argument("--output", type=str, default=OUTPUT_CSV, help="Output CSV file path")
    
    args = parser.parse_args()
    main(args)
