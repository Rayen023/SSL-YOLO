import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from PIL import Image
from pathlib import Path

import albumentations as A
from pytorch_metric_learning.losses import NTXentLoss

import torchvision
torchvision.disable_beta_transforms_warning()

from torchvision.transforms import v2
import os
from torch.utils.data import DataLoader, Dataset
#from torchvision import transforms as T
import matplotlib.pyplot as plt
import numpy as np
import tqdm



batch_size = 180
num_epochs = 1000
img_size = 224

print(f'batch_size : {batch_size} num_epochs : {num_epochs} img_size : {img_size}')

class ImageFolderCustom(Dataset):
    
    
    def __init__(self, targ_dir: str, transform=None) -> None:
        self.paths = list(sorted(Path(targ_dir).glob("*.jpg"))) 
        self.transform = transform

    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path) 
    
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    
    def __getitem__(self, index: int):
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index) # load image

        if img.mode != "RGB":
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img


data_transform = v2.Compose([
    v2.Resize(size=(img_size, img_size)),
    v2.ToImageTensor(),
    v2.ConvertImageDtype(dtype = torch.uint8),
])

root = "/gpfs/scratch/rayen/datasets/steel-common-aug/images/"
train_dir = os.path.join(root, "train/")
print(f"train_dir: {train_dir}")

train_data = ImageFolderCustom(
    targ_dir=train_dir,
    transform=data_transform,
    )

print(f"Length of train_data: {len(train_data)}")

train_dataloader = DataLoader(
    dataset=train_data, 
    batch_size=batch_size, 
    num_workers=20, 
    shuffle=True
    ) 
print(f"Length of train_dataloader: {len(train_dataloader)}")

imgs = next(iter(train_dataloader))

trained_layers = 11 

model = YOLO("yolov8l.yaml")  # build a new model from scratch

model_children_list = list(model.model.children())
backbone = model_children_list[0][:trained_layers]


augmentation = v2.Compose([
    v2.RandomResizedCrop(size=img_size, scale=(0.08, 1.0)),
    #v2.RandomEqualize(p=1.0),
    torchvision.transforms.functional.equalize,
    v2.ColorJitter(brightness=0.2, contrast=0.2),
    v2.RandomRotation(degrees=90),
    v2.RandomAffine(degrees=0, translate=(0, 0.1)),
    v2.GaussianBlur(kernel_size=(9, 9)),
    v2.ConvertImageDtype(),
])

"""augmentation = v2.Compose([
    v2.RandomResizedCrop(size=(img_size, img_size) , antialias = True),  # Or Resize(antialias=True)
    v2.RandomRotation(degrees=90),
    v2.RandomEqualize(p=0.8),
    v2.GaussianBlur(kernel_size=(9, 9)),
])"""

"""albumentations_transform = A.Compose([
    A.Rotate(limit=90, p=1.0),
    A.CLAHE(p=1.0),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.ShiftScaleRotate(shift_limit_y=0.1, scale_limit=0, rotate_limit=0, p=1.0)
])
"""

# Defining Model
class SimYOLOv8(nn.Module):
    def __init__(self):
        super().__init__()
        # Feature extraction
        self.backbone = backbone
        # Projection head
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # This will make the output (batch_size, 512, 1, 1)
            nn.Flatten(),  # This will make the output (batch_size, 512)
            nn.Linear(512, 256),
        )

    def forward(self, x, train=True):
        if train:
            # Get 2 augmentations of the batch
            augm_1 = augmentation(x)
            augm_2 = augmentation(x)

            # Get representations for first augmented view
            h_1 = self.backbone(augm_1)

            # Get representations for second augmented view
            h_2 = self.backbone(augm_2)
        else:
            h = self.backbone(x)
            return h

        # Transformation for loss function
        compact_h_1 = self.mlp(h_1)
        compact_h_2 = self.mlp(h_2)
        return h_1, h_2, compact_h_1, compact_h_2

    
# InfoNCE Noise-Contrastive Estimation
loss_func = NTXentLoss(temperature=0.25)

# higher batch sizes return better results usually from 256 to 8192 etc
# for batch size 1024, we get 1022 negative samples to model contrast against within a batch + our poisitive pair

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

assert str(device) == 'cuda' 
model = SimYOLOv8()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for _, data in enumerate(train_dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        # Get data representations
        h_1, h_2, compact_h_1, compact_h_2 = model(data)
        # Prepare for loss
        embeddings = torch.cat((compact_h_1, compact_h_2), dim = 0)
        # The same index corresponds to a positive pair
        indices = torch.arange(0, compact_h_1.size(0), device=compact_h_2.device)
        labels = torch.cat((indices, indices))
        loss = loss_func(embeddings, labels)
        loss.backward()
        total_loss += loss.item() * data.size(0)  
        optimizer.step()

    loss = total_loss / len(train_data) 
    print(f'Epoch {epoch:03d} / {num_epochs}, Loss: {loss:.4f}')
    scheduler.step()
    
    
# Extracting Backbone
backbone = model.backbone
print(backbone , backbone.state_dict())

model = YOLO("yolov8l.yaml")  # build a new model from scratch
model_children_list = list(model.model.children())
head_layers = model_children_list[0][trained_layers:]

full_state_dict = {**backbone.state_dict(), **head_layers.state_dict()}
full_state_dict = {f'model.{k}': v for k, v in full_state_dict.items()}

torch.save(full_state_dict, "yolov8l_back_steel.pt")


model = YOLO("yolov8l.yaml") 

model.train(data="/gpfs/scratch/rayen/datasets/steel-fs-aug/neu_det.yaml", epochs=300, batch=64, imgsz=224, device=0, pretrained = 'yolov8l_back_steel.pt')