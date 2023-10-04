import torch
import torch.nn as nn
from ultralytics import YOLO

trained_layers = 11  # specify how many pretrained layers to load

trained_model = YOLO("/gpfs/scratch/rayen/YOLOv8/yolov8l.pt")
trained_model_children_list = list(trained_model.model.children())
backbone = trained_model_children_list[0][:trained_layers]

#print(trained_model_children_list)

#for i, module in enumerate(trained_model.model.modules()): print(f"Module {i}:", module)

model = YOLO("yolov8l.yaml")  # build a new model from scratch
model_children_list = list(model.model.children())
head_layers = model_children_list[0][trained_layers:]

full_state_dict = {**backbone.state_dict(), **head_layers.state_dict()}
full_state_dict = {f'model.{k}': v for k, v in full_state_dict.items()}

torch.save(full_state_dict, "yolov8l_coco.pt")
