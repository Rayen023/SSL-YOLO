# SSL-YOLO

A self-supervised approach for few-shot object detection using contrastive learning with YOLOv8.

![Pipeline Visualization](images/pipeline.png)

## Description

SSL-YOLO is a project that employs a self-supervised approach to pretrain the backbone of YOLOv8 models for few-shot object detection. The framework uses contrastive representation learning to learn meaningful feature representations from unlabeled data before fine-tuning on a small labeled dataset.

The two-phase training process consists of:

1. **Self-Supervised Contrastive Learning Phase**: Pretrains the backbone using unlabeled images to learn general visual representations.
2. **Supervised Object Detection Phase**: Fine-tunes the model for object detection using a small labeled dataset.

This approach improves performance in scenarios where labeled data is scarce but unlabeled data is abundant.

## Features

- Self-supervised pretraining using contrastive learning
- Support for YOLOv8 model variants (n, s, m, l, x)
- Few-shot object detection capability
- Customizable data augmentation pipeline
- Compatible with the Ultralytics YOLOv8 framework

## Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (training is not feasible on CPU)
- Ultralytics v8.0.117 (modified version, adjusted the `ultralytics/yolo/engine/trainer.py` file to enable loading and freezing of the pretrained backbone)

## Installation

```bash
# Clone the repository
git clone https://github.com/Rayen023/ssl-yolo.git
cd ssl-yolo

# Install dependencies
pip install -r requirements.txt
```

## Setup & Configuration

### 1. Prepare Datasets

#### Self-Supervised Learning Dataset
- Collect a large dataset of unlabeled images related to your domain
- Place the images in a directory structure as specified in the configuration

#### Few-Shot Object Detection Dataset
- Prepare a small dataset (approximately 10 images per class) in YOLOv8 format
- Create a YAML configuration file specifying the dataset structure

### 2. Configuration Settings

Adjust the following parameters in the script according to your needs:

```python
# === Dataset Parameters ===
SSL_TRAIN_DIR = "/path/to/your/unlabeled/dataset/"
DET_DATASET_PATH = "/path/to/your/detection/dataset.yaml"

# === Shared Parameters ===
MODEL_YAML = "yolov8l.yaml"  # Change based on model size (n, s, m, l, x)
IMG_SIZE = 224
BACKBONE_LAYERS = 11  # Adjust based on model variant

# === Self-Supervised Learning Parameters ===
SSL_BATCH_SIZE = 180  # Larger batch size recommended for contrastive learning
SSL_NUM_EPOCHS = 1000
SSL_LEARNING_RATE = 0.001

# === Object Detection Parameters ===
DET_BATCH_SIZE = 64
DET_NUM_EPOCHS = 300
```

### 4. Update Model Configuration

Ensure the number of classes in the YOLOv8 configuration file matches your dataset:

```bash
# Modify the number of classes (nc) parameter in:
# /ultralytics/models/v8/yolov8.yaml
```

## Usage

### 1. Run Self-Supervised Pretraining

```bash
python ssl_training.py
```

This script will:
- Load your unlabeled dataset
- Create two augmented views of each image
- Train the backbone using contrastive learning
- Save the pretrained backbone weights to `ssl_pretrained_backbone_weights.pt`

For optimal results:
- Use a larger batch size to increase negative samples
- Train for more epochs to achieve better representations
- Adjust augmentation strategies based on your domain

### 2. Run Supervised Detection Training

```bash
# The pretraining script automatically runs the supervised training phase
# If you want to run it separately:
python utils/train_detector.py
```

This script will:
- Load the pretrained backbone weights
- Freeze the backbone layers
- Fine-tune the model on your few-shot dataset
- Save the resulting model

## How It Works

### Contrastive Learning Phase

1. **Data Loading**: Unlabeled images are loaded and prepared for training
2. **Augmentation**: Each image undergoes two different random augmentations
3. **Feature Extraction**: Both augmented versions pass through the backbone
4. **Projection**: Features are projected to a lower-dimensional space
5. **Contrastive Loss**: NT-Xent loss pushes together features from the same image and pulls apart features from different images
6. **Optimization**: The backbone learns to extract meaningful representations

### Object Detection Phase

1. **Backbone Loading**: The pretrained backbone is loaded into a YOLOv8 model
2. **Fine-tuning**: The model is trained on a small labeled dataset (10-shot)
3. **Evaluation**: The model is evaluated on the test set

## Tips for Best Results

1. **Dataset Selection**: Use an unlabeled dataset that is contextually similar to your target domain
2. **Augmentation Strategy**: Customize the data augmentations based on your specific use case
3. **Batch Size**: Use the largest batch size your GPU memory allows for better contrastive learning
4. **Training Duration**: Longer pretraining generally leads to better representations
5. **Learning Rate Scheduling**: Adjust the learning rate schedule for optimal convergence
<!--
## Citation

If you use this code for your research, please cite:

```
@misc{ssl-yolo,
  author = {Your Name},
  title = {SSL-YOLO: Self-Supervised Learning for Few-Shot Object Detection with YOLOv8},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/ssl-yolo}}
}
```

## License

[MIT License](LICENSE)
-->
## Acknowledgements

- This project is based on the [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) implementation
- The contrastive learning approach is inspired by [SimCLR](https://arxiv.org/abs/2002.05709)