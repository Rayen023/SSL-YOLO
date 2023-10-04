import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from PIL import Image
from pathlib import Path

import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import matplotlib.pyplot as plt
import numpy as np
import tqdm


print(torch.cuda.is_available())

# 1. Subclass torch.utils.data.Dataset
class ImageFolderCustom(Dataset):
    
    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str, transform=None) -> None:
        
        # 3. Create class attributes
        # Get all image paths
        self.paths = list(sorted(Path(targ_dir).glob("*.jpg"))) # note: you'd have to update this if you've got .png's or .jpeg's
        # Setup transforms
        self.transform = transform

    # 4. Make function to load images
    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path) 
    
    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    
    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int):
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index) # load image

        # Convert image to RGB if not
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Transform if necessary
        if self.transform:
            img = self.transform(img)

        return img


data_transform = T.Compose([
    # Resize the images to 64x64
    T.Resize(size=(544, 544)),
    # Flip the images randomly on the horizontal
    #transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
    # Turn the image into a torch.Tensor
    T.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
])
root = "/gpfs/scratch/rayen/10-shot_coco/coco/"
#root = "/gpfs/scratch/rayen/YOLOv8/wood_dataset/"
train_dir = os.path.join(root, "train2017/")
print(f"train_dir: {train_dir}")
train_data = ImageFolderCustom(targ_dir=train_dir, # target folder of images
                                  transform=data_transform, #data_transform, # transforms to perform on data (images)
                                    )
print(f"Length of train_data: {len(train_data)}")
train_dataloader = DataLoader(dataset=train_data, 
                              batch_size=102, #how many samples per batch?
                              num_workers=20, # how many subprocesses to use for data loading? (higher = more)
                              shuffle=True) # shuffle the data?

print(f"Length of train_dataloader: {len(train_dataloader)}")

imgs = next(iter(train_dataloader))
#print(f"Image shape: {imgs.shape}") # (batch_size, channels, height, width)

def plot_imgs(batch_imgs: torch.Tensor) -> None:
    "Plots all images in a batch."
    batch_size = batch_imgs.shape[0]
    rows = int(np.sqrt(batch_size))
    cols = int(np.ceil(batch_size / rows))
    
    fig, axs = plt.subplots(rows, cols, figsize=(15, 15))

    for i, img in enumerate(batch_imgs):
        ax = axs[i // cols, i % cols]
        ax.imshow(img.permute(1, 2, 0)) # permute to (height, width, channels)
        ax.axis("off")

    # If the number of images in the batch does not perfectly fill the grid, remove the empty subplots
    if batch_size < rows * cols:
        for ax in axs.flat[batch_size:]:
            ax.remove()
    
    plt.tight_layout()
    plt.show()

#plot_imgs(imgs)


# Extrating Backbone
trained_layers = 11  # specify how many pretrained layers to load

model = YOLO("yolov8l.yaml")  # build a new model from scratch

model_children_list = list(model.model.children())
backbone = model_children_list[0][:trained_layers]


# Plot torch model
#from torchviz import make_dot
#yhat = backbone(imgs ) # Give dummy batch to forward().
#make_dot(yhat, params=dict(list(backbone.named_parameters()))).render("rnn_torchviz", format="png")
#print(f"yhat shape : {yhat.shape}")



augmentation = T.Compose([
    T.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.RandomRotation(degrees=90),
    T.RandomAffine(degrees=0, translate=(0, 0.1)),
    T.GaussianBlur(kernel_size=(9, 9)),
])





"""augmentation = T.Compose([
    # Add color jitter
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # Randomly change the brightness, contrast and saturation
    T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)], p=0.8),
    # Randomly convert to grayscale
    T.RandomGrayscale(p=0.1),
    # Add Gaussian Blur
    T.GaussianBlur(kernel_size=(9, 9)),  # assuming image size is 224
    # Rotate the image by a few degrees
    T.RandomRotation(degrees=15),
    # Random horizontal flipping
    T.RandomHorizontalFlip(p=0.5),
])"""


# Defining Model
class Model(nn.Module):
    def __init__(self, k=20, aggr='max'):
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
from pytorch_metric_learning.losses import NTXentLoss
loss_func = NTXentLoss(temperature=0.25)

# higher batch sizes return better results usually from 256 to 8192 etc
# exemple : for batch size 1024, we get 1022 negative samples to model contrast against within a batch + our poisitive pair

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
assert str(device) == 'cuda' 
model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

def train():
    model.train()
    total_loss = 0
    for _, data in enumerate(tqdm.tqdm(train_dataloader)):
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
    return total_loss / len(train_data)  


for epoch in range(1, 40):
    loss = train()
    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')
    scheduler.step()
    
    
# Extracting Backbone
backbone = model.backbone
print(backbone , backbone.state_dict())

model = YOLO("yolov8l.yaml")  # build a new model from scratch
model_children_list = list(model.model.children())
head_layers = model_children_list[0][trained_layers:]

full_state_dict = {**backbone.state_dict(), **head_layers.state_dict()}
full_state_dict = {f'model.{k}': v for k, v in full_state_dict.items()}

torch.save(full_state_dict, "yolov8l_back_coco.pt")

print("pretrained model saved")

