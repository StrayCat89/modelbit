import modelbit, sys
from typing import *
from functools import cache
from transformers.models.llava.modeling_llava import LlavaForConditionalGeneration
from transformers.models.auto.processing_auto import AutoProcessor
import PIL.Image as Image
import requests
import modelbit as mb

@cache
def load_model():
  model = LlavaForConditionalGeneration.from_pretrained("./llava-hf", local_files_only=True, load_in_8bit=True)
  processor = AutoProcessor.from_pretrained("./llava-hf", local_files_only=True, load_in_8bit=True)
  return model, processor


# main function
def prompt_llava(url: str, prompt: str):
  model, processor = load_model()
  image = Image.open(requests.get(url, stream=True).raw)
  mb.log_image(image) # Log the input image in Modelbit
  full_prompt = f"USER: <image>\n{prompt} ASSISTANT:"
  inputs = processor(text=full_prompt, images=image, return_tensors="pt").to("cuda")
  generate_ids = model.generate(**inputs, max_new_tokens=15)
  response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].split("ASSISTANT:")[1]
  return response

# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   result = prompt_llava(...)
#   print(result)