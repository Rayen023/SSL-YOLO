from ultralytics import YOLO
from ssl_training import MODEL_YAML, DET_DATASET_PATH, DET_NUM_EPOCHS, DET_BATCH_SIZE, IMG_SIZE, DET_DEVICE, PRETRAINED_BACKBONE_PATH, DET_SAVE_DIRECTORY

model = YOLO(MODEL_YAML)


model.train(
    data=DET_DATASET_PATH, 
    epochs=DET_NUM_EPOCHS, 
    batch=64, 
    imgsz=320, 
    device=DET_DEVICE, 
    #pretrained=PRETRAINED_BACKBONE_PATH,
    project=DET_SAVE_DIRECTORY,  
)
metrics = model.val()  
print(f"Validation metrics: {metrics}")
