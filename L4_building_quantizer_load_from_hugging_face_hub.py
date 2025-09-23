import torch

from helper import W8A16LinearLayer
from helper import replace_linear_with_target_and_quantize, replace_linear_with_target
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import OPTForCausalLM, AutoTokenizer, AutoConfig
from transformers import pipeline
from huggingface_hub import hf_hub_download


## ------------------------------------------------------##
model_id = "./models/facebook/opt-125m"

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype = torch.bfloat16,
                                             low_cpu_mem_usage = True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

## ------------------------------------------------------##
replace_linear_with_target_and_quantize(model, W8A16LinearLayer, ["lm_head"])

## ------------------------------------------------------##
model   # print()

## ------------------------------------------------------##
quantized_state_dict = model.state_dict()
torch.save(quantized_state_dict, "quantized_state_dict.pth")

## ------------------------------------------------------##
model_id = "./models/facebook/opt-125m"
config = AutoConfig.from_pretrained(model_id)

with torch.device("meta"):
  model = OPTForCausalLM(config)

tokenizer = AutoTokenizer.from_pretrained(model_id)

## ------------------------------------------------------##
for param in model.parameters():
   print(param)

## ------------------------------------------------------##
model   # print()

## ------------------------------------------------------##
replace_linear_with_target(model, W8A16LinearLayer, ["lm_head"])

## ------------------------------------------------------##
model   # print()

## ------------------------------------------------------##
state_dict_cache_path = hf_hub_download("ybelkada/opt-125m-quantized-dlai",
                                        "quantized_state_dict.pth")

## ------------------------------------------------------##
state_dict = torch.load(state_dict_cache_path)

## ------------------------------------------------------##
model.load_state_dict(state_dict, strict = True, assign = True)

## ------------------------------------------------------##
pipe = pipeline("text-generation", model = model, tokenizer = tokenizer)
pipe("Hello today I am", max_new_tokens = 40)

## ------------------------------------------------------##
pipe = pipeline("text-generation", model = model, tokenizer = tokenizer)
pipe("Hello today I am giving a course about", max_new_tokens = 10)
