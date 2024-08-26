from ultralytics import YOLO

model = YOLO("yolov8l.pt")

def yolo_for_object_detection(url: str):
  results = model(url)
  return results[0].boxes.xyxy.to('cpu').numpy()