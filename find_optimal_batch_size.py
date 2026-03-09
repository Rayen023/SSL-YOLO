import os
import csv
import torch
import torch.nn as nn
from ultralytics import YOLO
import gc
from pathlib import Path
import argparse
from tqdm import tqdm
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pytorch_metric_learning.losses import NTXentLoss


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

YOLO_MODELS = [
    "yolov8m.yaml",
    "yolov8l.yaml",
    "yolov8x.yaml",
]

# Image sizes to test (multiples of 32 for YOLO)
IMAGE_SIZES = [224, 320, 416, 512, 640]

# Initial batch sizes to test
INITIAL_BATCH_SIZES = {"yolov8m.yaml": 32, "yolov8l.yaml": 16, "yolov8x.yaml": 8}

BACKBONE_LAYERS = 11

OUTPUT_CSV = "batch_size_results.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAINING_EPOCHS = 2


class SimpleImageDataset(Dataset):
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


def reset_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()


class SimYOLOv8ForTesting(nn.Module):
    def __init__(self, backbone, img_size, model_yaml):
        super().__init__()
        self.backbone = backbone
        model_type = Path(model_yaml).stem

        # Determine input features based on the model YAML name
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

        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_features, bias=True),
            nn.BatchNorm1d(out_features, affine=False),
        )

        self.transform = v2.Compose(
            [
                v2.RandomResizedCrop(size=img_size, scale=(0.7, 1.0)),
                v2.ColorJitter(brightness=0.1, contrast=0.2),
                v2.RandomRotation(degrees=30),
                v2.ConvertImageDtype(),
            ]
        )

    def forward(self, x):
        augm_1 = self.transform(x)
        augm_2 = self.transform(x)

        h_1 = self.backbone(augm_1)
        h_2 = self.backbone(augm_2)

        return self.mlp(h_1), self.mlp(h_2)


def create_yolo_backbone(model_yaml):
    model = YOLO(model_yaml)
    return list(model.model.children())[0][:BACKBONE_LAYERS]


def test_configuration(model_yaml, img_size, batch_size, img_dir):
    try:
        reset_gpu_memory()

        transform = v2.Compose(
            [
                v2.Resize(size=(img_size, img_size)),
                v2.ToImageTensor(),
                v2.ConvertImageDtype(dtype=torch.float32),
            ]
        )

        dataset = SimpleImageDataset(img_dir, transform=transform, max_images=200)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, num_workers=4, shuffle=True
        )

        backbone = create_yolo_backbone(model_yaml)
        model = SimYOLOv8ForTesting(backbone, img_size, model_yaml).to(DEVICE)
        model.train()

        loss_fn = NTXentLoss(temperature=0.25)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(TRAINING_EPOCHS):
            for data in dataloader:
                data = data.to(DEVICE)
                optimizer.zero_grad()

                emb1, emb2 = model(data)
                embeddings = torch.cat([emb1, emb2], dim=0)

                actual_batch_size = emb1.size(0)
                indices = torch.arange(actual_batch_size, device=DEVICE)
                labels = torch.cat([indices, indices])

                loss = loss_fn(embeddings, labels)
                loss.backward()
                optimizer.step()

        del model, backbone, optimizer, embeddings, loss
        reset_gpu_memory()

        return {
            "model": Path(model_yaml).stem,
            "image_size": img_size,
            "batch_size": batch_size,
            "status": "success",
        }

    except RuntimeError as e:
        reset_gpu_memory()
        if "out of memory" in str(e):
            return {
                "model": Path(model_yaml).stem,
                "image_size": img_size,
                "batch_size": batch_size,
                "status": "OOM",
            }
        return {
            "model": Path(model_yaml).stem,
            "image_size": img_size,
            "batch_size": batch_size,
            "status": f"error: {str(e)[:100]}...",
        }


def find_max_batch_size(model_yaml, img_size, img_dir):
    initial_batch = INITIAL_BATCH_SIZES.get(model_yaml, 8)
    min_batch = 1
    max_batch = initial_batch

    result = test_configuration(model_yaml, img_size, initial_batch, img_dir)
    if result["status"] != "success":
        max_batch = max(2, initial_batch // 2)
    else:
        while True:
            next_batch = max_batch + max(1, max_batch // 4)
            if (
                test_configuration(model_yaml, img_size, next_batch, img_dir)["status"]
                != "success"
                or next_batch >= 512
            ):
                break
            max_batch = next_batch

    while max_batch - min_batch > 1:
        mid_batch = (min_batch + max_batch) // 2
        if (
            test_configuration(model_yaml, img_size, mid_batch, img_dir)["status"]
            == "success"
        ):
            min_batch = mid_batch
        else:
            max_batch = mid_batch

    final_result = test_configuration(model_yaml, img_size, min_batch, img_dir)
    if final_result["status"] == "success":
        return final_result
    if min_batch > 1:
        return test_configuration(model_yaml, img_size, 1, img_dir)
    return final_result


def main(args):
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This script requires a GPU.")
        return

    results = []
    for model_yaml in tqdm(YOLO_MODELS, desc="Testing models"):
        for img_size in tqdm(
            IMAGE_SIZES, desc=f"Testing {model_yaml} image sizes", leave=False
        ):
            results.append(find_max_batch_size(model_yaml, img_size, args.img_dir))

    with open(args.output, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=["model", "image_size", "batch_size", "status"]
        )
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default=SSL_TRAIN_DIR)
    parser.add_argument("--output", type=str, default=OUTPUT_CSV)
    main(parser.parse_args())
