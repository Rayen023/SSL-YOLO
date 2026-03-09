import os
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from dotenv import load_dotenv

import torchvision

torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2

from pytorch_metric_learning.losses import NTXentLoss
from ultralytics import YOLO

load_dotenv()


def load_config():
    app_env = os.environ.get("APP_ENV", "")
    config_path = Path(__file__).parent / "config.yaml"

    if app_env == "local":
        local_config_path = Path(__file__).parent / "config-local.yaml"
        if local_config_path.exists():
            config_path = local_config_path

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


config = load_config()

SSL_TRAIN_DIR = config["datasets"]["ssl_train_dir"]
DET_DATASET_PATH = config["datasets"]["det_dataset_path"]

MODEL_YAML = config["model"]["yaml"]
IMG_SIZE = config["model"]["img_size"]
BACKBONE_LAYERS = config["model"]["backbone_layers"]

SSL_BATCH_SIZE = config["ssl"]["batch_size"]
SSL_NUM_EPOCHS = config["ssl"]["num_epochs"]
SSL_NUM_WORKERS = config["ssl"]["num_workers"]
SSL_LEARNING_RATE = config["ssl"]["learning_rate"]
SSL_SCHEDULER_STEP_SIZE = config["ssl"]["scheduler_step_size"]
SSL_SCHEDULER_GAMMA = config["ssl"]["scheduler_gamma"]
SSL_CONTRASTIVE_TEMP = config["ssl"]["contrastive_temp"]
SSL_PATIENCE = config["ssl"]["patience"]

DET_BATCH_SIZE = config["detection"]["batch_size"]
DET_NUM_EPOCHS = config["detection"]["num_epochs"]
DET_DEVICE = config["detection"]["device"]

ssl_dataset_name = (
    Path(SSL_TRAIN_DIR).parts[-3] if len(Path(SSL_TRAIN_DIR).parts) >= 3 else "dataset"
)
det_dataset_name = Path(DET_DATASET_PATH).stem
model_type = Path(MODEL_YAML).stem

hyperparams_suffix = (
    f"lr{SSL_LEARNING_RATE}_temp{SSL_CONTRASTIVE_TEMP}_sz{IMG_SIZE}_l{BACKBONE_LAYERS}"
)
backbone_filename = f"backbone_{ssl_dataset_name}_{model_type}_{hyperparams_suffix}.pt"
detector_dirname = (
    f"{model_type}_{det_dataset_name}_img{IMG_SIZE}_layers{BACKBONE_LAYERS}"
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_base_dir = f"train/results_{timestamp}"

PRETRAINED_BACKBONE_PATH = os.path.join(results_base_dir, backbone_filename)
DET_SAVE_DIRECTORY = os.path.join(results_base_dir, detector_dirname)

os.makedirs(os.path.dirname(PRETRAINED_BACKBONE_PATH), exist_ok=True)
os.makedirs(DET_SAVE_DIRECTORY, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageFolderCustom(Dataset):
    def __init__(self, targ_dir, transform=None):
        self.paths = list(sorted(Path(targ_dir).glob("*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = Image.open(self.paths[index])
        if img.mode != "RGB":
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def create_data_transforms():
    data_transform = v2.Compose(
        [
            v2.Resize(size=(IMG_SIZE, IMG_SIZE)),
            v2.ToImageTensor(),
            v2.ConvertImageDtype(dtype=torch.uint8),
        ]
    )

    augmentation = v2.Compose(
        [
            v2.RandomResizedCrop(size=IMG_SIZE, scale=(0.7, 1.0)),
            torchvision.transforms.functional.equalize,
            v2.ColorJitter(brightness=0.1, contrast=0.2),
            v2.RandomRotation(degrees=30),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.3),
            v2.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            v2.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0)),
            v2.RandomPerspective(distortion_scale=0.2, p=0.3),
            v2.RandomGrayscale(p=0.2),
            v2.ConvertImageDtype(),
        ]
    )

    return data_transform, augmentation


def load_data(data_transform):
    train_data = ImageFolderCustom(targ_dir=SSL_TRAIN_DIR, transform=data_transform)
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=SSL_BATCH_SIZE,
        num_workers=SSL_NUM_WORKERS,
        shuffle=True,
    )
    return train_data, train_dataloader


class SimYOLOv8(nn.Module):
    def __init__(self, backbone, augmentation):
        super().__init__()
        self.backbone = backbone
        self.augmentation = augmentation

        # Determine input features based on model type
        model_type = MODEL_YAML.split(".")[0]
        if "yolov8x" in model_type:
            in_features = 640
        elif "yolov8l" in model_type:
            in_features = 512
        elif "yolov8m" in model_type:
            in_features = 576
        elif "yolov8s" in model_type:
            in_features = 512
        elif "yolov8n" in model_type:
            in_features = 256
        else:
            in_features = 512

        out_features = 256
        hidden_dim = out_features * 2

        # Projection head: 2-layer MLP with BatchNorm and GELU
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_features, bias=True),
            nn.BatchNorm1d(out_features, affine=False),
        )

    def forward(self, x, train=True):
        if train:
            augm_1 = self.augmentation(x)
            augm_2 = self.augmentation(x)

            h_1 = self.backbone(augm_1)
            h_2 = self.backbone(augm_2)

            return h_1, h_2, self.mlp(h_1), self.mlp(h_2)
        return self.backbone(x)


def create_yolo_backbone():
    model = YOLO(MODEL_YAML)
    model_children_list = list(model.model.children())
    backbone = model_children_list[0][:BACKBONE_LAYERS]
    return model, backbone, model_children_list


def train_contrastive(
    model, train_dataloader, train_data, optimizer, scheduler, loss_func
):
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(SSL_NUM_EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{SSL_NUM_EPOCHS}")

        for data in progress_bar:
            data = data.to(DEVICE)
            optimizer.zero_grad()

            _, _, compact_h_1, compact_h_2 = model(data)
            embeddings = torch.cat((compact_h_1, compact_h_2), dim=0)

            indices = torch.arange(0, compact_h_1.size(0), device=compact_h_2.device)
            labels = torch.cat((indices, indices))

            loss = loss_func(embeddings, labels)
            loss.backward()
            total_loss += loss.item() * data.size(0)
            optimizer.step()

            progress_bar.set_postfix({"batch_loss": loss.item()})

        epoch_loss = total_loss / len(train_data)
        print(f"Epoch {epoch+1:03d}/{SSL_NUM_EPOCHS}, Loss: {epoch_loss:.4f}")
        scheduler.step()

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                },
                PRETRAINED_BACKBONE_PATH,
            )
        else:
            patience_counter += 1

        if patience_counter >= SSL_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    if os.path.exists(PRETRAINED_BACKBONE_PATH):
        checkpoint = torch.load(PRETRAINED_BACKBONE_PATH)
        model.load_state_dict(checkpoint["model_state_dict"])


def extract_backbone_and_save(model, backbone):
    yolo_model = YOLO(MODEL_YAML)
    head_layers = list(yolo_model.model.children())[0][BACKBONE_LAYERS:]

    full_state_dict = {
        f"model.{k}": v
        for k, v in {**backbone.state_dict(), **head_layers.state_dict()}.items()
    }
    torch.save(full_state_dict, PRETRAINED_BACKBONE_PATH)


def train_detector():
    model = YOLO(MODEL_YAML)
    model.train(
        data=DET_DATASET_PATH,
        epochs=DET_NUM_EPOCHS,
        batch=DET_BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DET_DEVICE,
        pretrained=PRETRAINED_BACKBONE_PATH,
        project=DET_SAVE_DIRECTORY,
    )
    print(f"Validation metrics: {model.val()}")


def main():
    data_transform, augmentation = create_data_transforms()
    train_data, train_dataloader = load_data(data_transform)
    _, backbone, _ = create_yolo_backbone()

    model = SimYOLOv8(backbone, augmentation).to(DEVICE)
    loss_func = NTXentLoss(temperature=SSL_CONTRASTIVE_TEMP)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SSL_LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=SSL_SCHEDULER_STEP_SIZE, gamma=SSL_SCHEDULER_GAMMA
    )

    train_contrastive(
        model, train_dataloader, train_data, optimizer, scheduler, loss_func
    )
    extract_backbone_and_save(model, model.backbone)
    train_detector()


if __name__ == "__main__":
    assert str(DEVICE) == "cuda", "CUDA is required for training"
    main()
