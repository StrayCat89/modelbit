from ultralytics import SAM

model = SAM("sam2_b.pt")

def sam2_for_object_detection(url: str, x_coord: int, y_coord: int):
  results = model(url, points=[x_coord, y_coord], labels=[1])
  return results[0].masks.xy[0][0]