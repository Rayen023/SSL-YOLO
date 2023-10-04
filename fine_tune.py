from ultralytics import YOLO


model = YOLO("yolov8l.yaml") 


model.train(data="/gpfs/scratch/rayen/YOLOv8/datasets/aug-fs-metatrain/neu_det.yaml", epochs=300, batch=64, imgsz=224, device=0, pretrained = 'yolov8l_coco.pt')