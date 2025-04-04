import torch
import torch.nn as nn
from ultralytics import YOLO
from PIL import Image
import os
import yaml
from torch.utils.data import DataLoader, Dataset
import tqdm
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2

from pytorch_metric_learning.losses import NTXentLoss

# Load environment variables from .env file
load_dotenv()

# ====================================
# CONFIGURATION LOADING
# ====================================
def load_config():
    """Load configuration from YAML file based on environment"""
    app_env = os.environ.get('APP_ENV', '')
    
    # Default to config.yaml
    config_path = Path(__file__).parent / 'config.yaml'
    
    # Only use local config if explicitly set
    if app_env == 'local':
        local_config_path = Path(__file__).parent / 'config-local.yaml'
        if local_config_path.exists():
            config_path = local_config_path
            print(f"Using local configuration from {config_path}")
        else:
            print(f"Local configuration not found at {local_config_path}, using default")
    
    print(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

# Load configuration
config = load_config()

# ====================================
# GLOBAL CONFIGURATION
# ====================================

# Dataset paths
SSL_TRAIN_DIR = config['datasets']['ssl_train_dir']
DET_DATASET_PATH = config['datasets']['det_dataset_path']

# Model parameters
MODEL_YAML = config['model']['yaml']
IMG_SIZE = config['model']['img_size']
BACKBONE_LAYERS = config['model']['backbone_layers']

# Self-supervised learning parameters
SSL_BATCH_SIZE = config['ssl']['batch_size']
SSL_NUM_EPOCHS = config['ssl']['num_epochs']
SSL_NUM_WORKERS = config['ssl']['num_workers']
SSL_LEARNING_RATE = config['ssl']['learning_rate']
SSL_SCHEDULER_STEP_SIZE = config['ssl']['scheduler_step_size']
SSL_SCHEDULER_GAMMA = config['ssl']['scheduler_gamma']
SSL_CONTRASTIVE_TEMP = config['ssl']['contrastive_temp']
SSL_PATIENCE = config['ssl']['patience']

# Supervised detection parameters
DET_BATCH_SIZE = config['detection']['batch_size']
DET_NUM_EPOCHS = config['detection']['num_epochs']
DET_DEVICE = config['detection']['device']

# --------------------------------------
# OUTPUT PATHS CONFIGURATION
# --------------------------------------
# Extract dataset names from paths for naming
ssl_dataset_name = Path(SSL_TRAIN_DIR).parts[-3]  
det_dataset_name = Path(DET_DATASET_PATH).stem  
model_type = Path(MODEL_YAML).stem

# Create descriptive suffix with hyperparameters
hyperparams_suffix = f"lr{SSL_LEARNING_RATE}_temp{SSL_CONTRASTIVE_TEMP}_sz{IMG_SIZE}_l{BACKBONE_LAYERS}"

# Base filenames
backbone_filename = f"backbone_{ssl_dataset_name}_{model_type}_{hyperparams_suffix}.pt"
detector_dirname = f"{model_type}_{det_dataset_name}_img{IMG_SIZE}_layers{BACKBONE_LAYERS}"

# Create timestamped results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_base_dir = f"train/results_{timestamp}"
os.makedirs(results_base_dir, exist_ok=True)

# Final paths
PRETRAINED_BACKBONE_PATH = os.path.join(results_base_dir, backbone_filename)
DET_SAVE_DIRECTORY = os.path.join(results_base_dir, detector_dirname)

# Create directories
os.makedirs(os.path.dirname(PRETRAINED_BACKBONE_PATH), exist_ok=True)
os.makedirs(DET_SAVE_DIRECTORY, exist_ok=True)

# Print paths
print(f"Output paths: backbone={PRETRAINED_BACKBONE_PATH}, detector={DET_SAVE_DIRECTORY}, results={results_base_dir}")

# Print configuration sections
for section, params in config.items():
    if isinstance(params, dict):
        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        print(f"{section}: {param_str}")# Device configuration
        
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====================================
# DATASET AND DATA LOADING
# ====================================

class ImageFolderCustom(Dataset):
    """Custom dataset for loading images from a directory"""
    
    def __init__(self, targ_dir: str, transform=None) -> None:
        self.paths = list(sorted(Path(targ_dir).glob("*.jpg"))) # Adjust for other image formats
        self.transform = transform

    def load_image(self, index: int) -> Image.Image:
        """Opens an image via a path and returns it."""
        image_path = self.paths[index]
        return Image.open(image_path) 
    
    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.paths)
    
    def __getitem__(self, index: int):
        """Returns one sample of data"""
        img = self.load_image(index)

        if img.mode != "RGB":
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img

def create_data_transforms():
    """Create data transforms for the dataset"""
    data_transform = v2.Compose([
        v2.Resize(size=(IMG_SIZE, IMG_SIZE)),
        v2.ToImageTensor(),
        v2.ConvertImageDtype(dtype=torch.uint8),
    ])
    
    augmentation = v2.Compose([
        v2.RandomResizedCrop(size=IMG_SIZE, scale=(0.7, 1.0)),  # Less aggressive cropping to preserve defects
        torchvision.transforms.functional.equalize,  # Good for enhancing contrast
        v2.ColorJitter(brightness=0.1, contrast=0.2),  # Reduced brightness jitter
        v2.RandomRotation(degrees=30),  # Reduced rotation angle
        v2.RandomHorizontalFlip(p=0.5),  # Added for orientation invariance
        v2.RandomVerticalFlip(p=0.3),    # Added for orientation invariance
        v2.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # Reduced translation
        v2.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0)),  # Smaller, variable kernel
        v2.RandomPerspective(distortion_scale=0.2, p=0.3),  # Added perspective change
        # Occasionally convert to grayscale since texture is important
        v2.RandomGrayscale(p=0.2),
        # Add random noise occasionally
        # Note: you'll need to implement a custom noise transform
        v2.ConvertImageDtype(),
    ])
    
    return data_transform, augmentation


def load_data(data_transform):
    """Load and prepare data for self-supervised contrastive learning"""
    print(f"Train directory: {SSL_TRAIN_DIR}")
    
    train_data = ImageFolderCustom(
        targ_dir=SSL_TRAIN_DIR,
        transform=data_transform,
    )
    
    print(f"Length of train_data: {len(train_data)}")
    
    train_dataloader = DataLoader(
        dataset=train_data, 
        batch_size=SSL_BATCH_SIZE, 
        num_workers=SSL_NUM_WORKERS, 
        shuffle=True
    ) 
    
    print(f"Length of train_dataloader: {len(train_dataloader)}")
    
    return train_data, train_dataloader


# ====================================
# MODEL DEFINITION
# ====================================

class SimYOLOv8(nn.Module):
    """Contrastive learning model based on YOLOv8 backbone"""
    
    def __init__(self, backbone, augmentation):
        super().__init__()
        # Feature extraction
        self.backbone = backbone
        self.augmentation = augmentation
        
        # Dynamically determine input features based on model type
        model_type = MODEL_YAML.split('.')[0]
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
            # Default case or fallback
            in_features = 512
            print(f"Warning: Unknown model type '{model_type}'. Using default in_features={in_features}")
        
        # Improved projection head for better contrastive learning
        # - Uses 2-layer MLP with non-linearity in between (SimCLR approach)
        # - BatchNorm for better training stability
        # - Hidden dimension is 2x output dimension as per best practices
        # - GELU activation for better performance than ReLU in many cases
        out_features = 256  # projection dimension
        hidden_dim = out_features * 2
        
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_features, bias=True),
            nn.BatchNorm1d(out_features, affine=False)  # Normalize embeddings for cosine similarity
        )

    def forward(self, x, train=True):
        if train:
            # Get 2 augmentations of the batch
            augm_1 = self.augmentation(x)
            augm_2 = self.augmentation(x)

            # Get representations for first augmented view
            h_1 = self.backbone(augm_1)

            # Get representations for second augmented view
            h_2 = self.backbone(augm_2)
            
            # Transformation for loss function
            compact_h_1 = self.mlp(h_1)
            compact_h_2 = self.mlp(h_2)
            
            return h_1, h_2, compact_h_1, compact_h_2
        else:
            h = self.backbone(x)
            return h


def create_yolo_backbone():
    """Extract backbone layers from YOLOv8 model"""
    model = YOLO(MODEL_YAML)  # build a new model from scratch
    model_children_list = list(model.model.children())
    backbone = model_children_list[0][:BACKBONE_LAYERS]
    return model, backbone, model_children_list


# ====================================
# TRAINING FUNCTIONS
# ====================================

def train_contrastive(model, train_dataloader, train_data, optimizer, scheduler, loss_func):
    """Train the model using contrastive learning"""
    print(f"Starting contrastive learning training for {SSL_NUM_EPOCHS} epochs")
    
    # Setup for early stopping and model checkpointing
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(SSL_NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        # Add tqdm progress bar
        progress_bar = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{SSL_NUM_EPOCHS}")
        
        for _, data in enumerate(progress_bar):
            data = data.to(DEVICE)
            optimizer.zero_grad()
            
            # Get data representations
            h_1, h_2, compact_h_1, compact_h_2 = model(data)
            
            # Prepare for loss
            embeddings = torch.cat((compact_h_1, compact_h_2), dim=0)
            
            # The same index corresponds to a positive pair
            indices = torch.arange(0, compact_h_1.size(0), device=compact_h_2.device)
            labels = torch.cat((indices, indices))
            
            loss = loss_func(embeddings, labels)
            loss.backward()
            total_loss += loss.item() * data.size(0)
            optimizer.step()
            
            # Update progress bar with current loss
            progress_bar.set_postfix({"batch_loss": loss.item()})

        epoch_loss = total_loss / len(train_data)
        print(f'Epoch {epoch+1:03d}/{SSL_NUM_EPOCHS}, Loss: {epoch_loss:.4f}')
        scheduler.step()
        
        # Check if this is the best model so far
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            # Save the best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, PRETRAINED_BACKBONE_PATH)
            print(f"New best model saved with loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs. Best loss: {best_loss:.4f}")
            
        # Check for early stopping
        if patience_counter >= SSL_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs due to no improvement for {SSL_PATIENCE} epochs.")
            break
    
    # Load the best model before returning
    if os.path.exists(PRETRAINED_BACKBONE_PATH):
        checkpoint = torch.load(PRETRAINED_BACKBONE_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']+1} with loss {checkpoint['loss']:.4f}")
    
    print("Contrastive learning training complete!")


def extract_backbone_and_save(model, backbone):
    """Extract and save the trained backbone"""
    print("Extracting trained backbone...")
    
    # Get YOLOv8 head layers
    yolo_model = YOLO(MODEL_YAML)
    model_children = list(yolo_model.model.children())
    head_layers = model_children[0][BACKBONE_LAYERS:]
    
    # Combine trained backbone with original head layers
    full_state_dict = {**backbone.state_dict(), **head_layers.state_dict()}
    full_state_dict = {f'model.{k}': v for k, v in full_state_dict.items()}
    
    # Save the combined model
    torch.save(full_state_dict, PRETRAINED_BACKBONE_PATH)
    print(f"Saved combined model to {PRETRAINED_BACKBONE_PATH}")


def train_detector():
    """Train YOLO detector using the pretrained backbone"""
    print(f"Training YOLO detector for {DET_NUM_EPOCHS} epochs with pretrained backbone...")
    
    model = YOLO(MODEL_YAML)
    model.train(
        data=DET_DATASET_PATH, 
        epochs=DET_NUM_EPOCHS, 
        batch=DET_BATCH_SIZE, 
        imgsz=IMG_SIZE, 
        device=DET_DEVICE, 
        pretrained=PRETRAINED_BACKBONE_PATH,
        project=DET_SAVE_DIRECTORY  
    )
    metrics = model.val()  
    print(f"Validation metrics: {metrics}")
    
    print("YOLO detector training complete!")


# ====================================
# MAIN EXECUTION
# ====================================

def main():
    """Main execution function"""
   
    # Step 1: Prepare data and model for contrastive learning
    print("\n===== STEP 1: PREPARING DATA AND MODEL =====")
    data_transform, augmentation = create_data_transforms()
    train_data, train_dataloader = load_data(data_transform)
    
    _, backbone, _ = create_yolo_backbone()
    
    # Create model, loss function, optimizer and scheduler for self-supervised learning
    model = SimYOLOv8(backbone, augmentation)
    model = model.to(DEVICE)
    
    loss_func = NTXentLoss(temperature=SSL_CONTRASTIVE_TEMP)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SSL_LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=SSL_SCHEDULER_STEP_SIZE, 
        gamma=SSL_SCHEDULER_GAMMA
    )
    
    # Step 2: Train backbone with contrastive learning
    print("\n===== STEP 2: TRAINING BACKBONE WITH CONTRASTIVE LEARNING =====")
    train_contrastive(model, train_dataloader, train_data, optimizer, scheduler, loss_func)
    
    # Step 3: Extract backbone and train YOLO detector
    print("\n===== STEP 3: TRAINING YOLO DETECTOR WITH PRETRAINED BACKBONE =====")
    extract_backbone_and_save(model, model.backbone)
    train_detector()


if __name__ == "__main__":
    # Ensure CUDA is available
    assert str(DEVICE) == 'cuda', "CUDA is not available, but required for training"
    main()