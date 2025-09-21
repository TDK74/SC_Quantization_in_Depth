import torch
import torch.nn as nn
import torch.nn.functional as F
import requests

from helper import W8A16LinearLayer, replace_linear_with_target_and_quantize
from helper import plot_results
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image


## ------------------------------------------------------##
model_id = "./models/Salesforce/codegen-350M-mono"

model = AutoModelForCausalLM.from_pretrained(model_id,
                                             torch_dtype = torch.bfloat16,
                                             low_cpu_mem_usage = True)

tokenizer = AutoTokenizer.from_pretrained(model_id)

## ------------------------------------------------------##
pipe = pipeline("text-generation", model = model, tokenizer = tokenizer)

## ------------------------------------------------------##
print(pipe("def hello_world():", max_new_tokens = 20, do_sample = False))

## ------------------------------------------------------##
print("Model before:\n\n", model)

## ------------------------------------------------------##
replace_linear_with_target_and_quantize(model, W8A16LinearLayer, ["lm_head"])

## ------------------------------------------------------##
pipe.model

## ------------------------------------------------------##
print(pipe("def hello_world(): ", max_new_tokens = 20, do_sample = False)[0]["generated_text"])

## ------------------------------------------------------##
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

## ------------------------------------------------------##
previous_memory_footprint = model.get_memory_footprint()

## ------------------------------------------------------##
print("Footprint of the model in MBs: ", previous_memory_footprint / 1e+6)

## ------------------------------------------------------##
img_path = "dinner_with_friends.png"
image = Image.open(img_path).convert("RGB")
image

## ------------------------------------------------------##
inputs = processor(images = image, return_tensors = "pt")

with torch.no_grad():
  outputs = model(**inputs)

target_sizes = torch.tensor([image.size[ : : -1]])
results = processor.post_process_object_detection(outputs, target_sizes = target_sizes,
                                                  threshold = 0.9)[0]

## ------------------------------------------------------##
plot_results(model, image, results)

## ------------------------------------------------------##
model   # print()

## ------------------------------------------------------##
replace_linear_with_target_and_quantize(model, W8A16LinearLayer,
                                        ["0", "1", "2", "class_labels_classifier"])

## ------------------------------------------------------##
model   # print()

## ------------------------------------------------------##
inputs = processor(images = image, return_tensors = "pt")

with torch.no_grad():
  outputs = model(**inputs)

target_sizes = torch.tensor([image.size[ : : -1]])
results = processor.post_process_object_detection(outputs, target_sizes = target_sizes,
                                                  threshold = 0.9)[0]

## ------------------------------------------------------##
plot_results(model, image, results)

## ------------------------------------------------------##
new_footprint = model.get_memory_footprint()

## ------------------------------------------------------##
print("Footprint of the model in MBs: ", new_footprint / 1e+6)

## ------------------------------------------------------##
print("Memory saved in MBs: ", (previous_memory_footprint - new_footprint) / 1e+6)
