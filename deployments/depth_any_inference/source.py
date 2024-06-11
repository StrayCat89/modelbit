import modelbit, sys
from typing import *
from functools import cache
from huggingface_hub._snapshot_download import snapshot_download
from transformers.models.auto.image_processing_auto import AutoImageProcessor
from transformers.models.auto.modeling_auto import AutoModelForDepthEstimation
from torch import Tensor
import PIL.Image as Image
import requests
import torch
import numpy as np
import io
import base64

pixel_values = modelbit.load_value("data/pixel_values.pkl") # tensor([[[[1.2899, 1.2214, 1.2557, ..., 1.7694, 1.7694, 1.8037], [1.3070, 1.3242, 1.2899, ..., 1.7865, 1.8037, 1.7694], [1.4269, 1.3413, 1.3413, ..., 1.7865, 1.8379, 1.7523], ..., [1.3584, 1.5125, 1.5...
predicted_depth = modelbit.load_value("data/predicted_depth.pkl") # tensor([[[ 2.3043, 2.1127, 2.0877, ..., 12.4454, 12.5369, 9.6246], [ 2.2539, 2.2174, 2.1192, ..., 12.2459, 12.1250, 12.5340], [ 2.3273, 2.2369, 2.1380, ..., 12.3770, 12.3582, 12.6861], ..., [16.9373, ...

@cache
def get_depth_any_dino_v2_backbone():
    model_path = snapshot_download(repo_id="nielsr/depth-anything-small")
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModelForDepthEstimation.from_pretrained(model_path)
    return model, processor


# main function
def depth_any_inference(image_url):
  model, processor = get_depth_any_dino_v2_backbone()
  print("Model Backbone loaded")
  image = Image.open(requests.get(image_url, stream=True).raw)
  print("Image url loaded")
  pixel_values = processor(images=image, return_tensors="pt").pixel_values
  with torch.no_grad():
    outputs = model(pixel_values)
    predicted_depth = outputs.predicted_depth

  depth = torch.nn.functional.interpolate(predicted_depth.unsqueeze(1), size=image.size[::-1],
                                          mode="bicubic", align_corners=False,)
  output = depth.squeeze().cpu().numpy()
  formatted = (output * 255 / np.max(output)).astype("uint8")
  formatted_depth = Image.fromarray(formatted)
  # Convert the image to a BytesIO object
  image_bytes = io.BytesIO()
  formatted_depth.save(image_bytes, format='PNG')
  image_bytes = image_bytes.getvalue()

  # Encode the image bytes using base64
  base64_encoded_image = base64.b64encode(image_bytes).decode('utf-8')

  # Create a dictionary with the base64 encoded image
  response = {
      'image': base64_encoded_image
  }

  return response

# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   result = depth_any_inference(...)
#   print(result)