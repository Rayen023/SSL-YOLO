from ultralytics import YOLO


model = YOLO("yolov8l.yaml") 


model.train(data="/gpfs/scratch/rayen/datasets/steel-fs-aug/neu_det.yaml", epochs=300, batch=64, imgsz=224, device=0, pretrained = 'yolov8l_back_wood.pt')
metrics = model.val()  
print(f"Validation metrics: {metrics}")