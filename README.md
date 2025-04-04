# SSL-YOLO

A semi-supervised approach for few-shot object detection using contrastive learning with YOLOv8.

![Pipeline Visualization](images/pipeline.png)

## Description

SSL-YOLO is a project that employs a self-supervised approach to pretrain the backbone of YOLOv8 models for few-shot object detection. The framework uses contrastive representation learning to learn meaningful feature representations from unlabeled data before fine-tuning on a small labeled dataset.

- On NEU-DET dataset, mAP@50 improved by 4.0% (from 0.726 to 0.755) by pretraining the backbone with domain-specific unlabeled images.
- Applied in a Few-Shot Object Detection (FSOD) context using the FS-ND dataset, it achieves a significant improvement in mAP@50 from a baseline of 0.127 to 0.571 (a 349% increase).

The two-phase training process consists of:

1. **Self-Supervised Contrastive Learning Phase**: Pretrains the backbone using unlabeled images to learn general visual representations.
2. **Supervised Object Detection Phase**: Fine-tunes the model for object detection using a small labeled dataset.

This approach improves performance in scenarios where labeled data is scarce, but unlabeled data is abundant.

## Features

- Self-supervised pretraining using contrastive learning
- Support for YOLOv8 model variants (n, s, m, l, x)
- Few-shot object detection capability
- Customizable data augmentation pipeline
- Based on Ultralytics v8.0.117 framework (modified `ultralytics/yolo/engine/trainer.py` file to enable loading and freezing of the pretrained backbone)

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

- **Semi-Supervised Learning**: Collect unlabeled images related to your domain
- **Few-Shot Object Detection**: Prepare a small dataset (~10 images per class) in YOLOv8 format

### 2. Configuration Settings

All parameters are managed through the `config.yaml` file:

```yaml
# === Dataset Parameters ===
ssl_train_dir: "/path/to/your/unlabeled/dataset/"
det_dataset_path: "/path/to/your/detection/dataset.yaml"

# === Shared Parameters ===
model_yaml: "yolov8l.yaml"  # Change based on model size (n, s, m, l, x)
img_size: 224
backbone_layers: 11  # Adjust based on model variant

# === Semi-Supervised Learning Parameters ===
ssl_batch_size: 180  # Larger batch size recommended for contrastive learning
ssl_num_epochs: 1000
ssl_learning_rate: 0.001

# === Object Detection Parameters ===
det_batch_size: 64
det_num_epochs: 300
```

### 3. Update Model Configuration

Ensure the number of classes in the YOLOv8 configuration file matches your dataset:

```bash
# Modify the number of classes (nc) parameter in:
# /ultralytics/models/v8/yolov8.yaml
```

## Usage

```bash
python ssl_training.py
```

This script will:
- Train the backbone using contrastive learning on unlabeled data
- Save the pretrained backbone weights
- Fine-tune the model on your few-shot dataset with the backbone frozen
- Save the resulting model

## How It Works

### Contrastive Learning Phase

1. **Data Augmentation**: Each image undergoes two different random augmentations
2. **Feature Extraction & Projection**: Both augmented versions pass through the backbone and are projected to a lower-dimensional space
3. **Contrastive Loss**: NT-Xent loss pushes together features from the same image and pulls apart features from different images

### Object Detection Phase

1. **Backbone Transfer**: The pretrained backbone is loaded into a YOLOv8 model
2. **Fine-tuning**: The model is trained on a small labeled dataset (10-shot)
3. **Evaluation**: The model is evaluated on the test set

## Tips for Best Results

1. **Dataset Selection**: Use an unlabeled dataset contextually similar to your target domain
2. **Augmentation Strategy**: Customize based on your specific use case
3. **Batch Size**: Use the largest batch size your GPU memory allows
4. **Training Duration**: Longer pretraining generally leads to better representations
5. **Learning Rate Scheduling**: Adjust for optimal convergence

<!--
## Citation

If you use this code for your research, please cite:

```
@misc{ssl-yolo,
  author = {Your Name},
  title = {SSL-YOLO: Semi-Supervised Learning for Few-Shot Object Detection with YOLOv8},
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