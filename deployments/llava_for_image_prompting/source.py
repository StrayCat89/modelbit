import modelbit

from typing import *
from transformers.models.llava.modeling_llava import LlavaForConditionalGeneration
from transformers.models.auto.processing_auto import AutoProcessor
import PIL.Image as Image
import requests

model = LlavaForConditionalGeneration.from_pretrained("./llava-hf", local_files_only=True, load_in_8bit=True)
processor = AutoProcessor.from_pretrained("./llava-hf", local_files_only=True, load_in_8bit=True)

# main function
def llava_for_image_prompting(url: str, prompt: str):
  image = Image.open(requests.get(url, stream=True).raw)
  modelbit.log_image(image)
  full_prompt = f"USER: <image>\n{prompt} ASSISTANT:"
  inputs = processor(text=full_prompt, images=image, return_tensors="pt").to("cuda")
  generate_ids = model.generate(**inputs, max_new_tokens=15)
  response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].split("ASSISTANT:")[1]
  return response