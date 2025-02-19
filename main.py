import numpy as np
import pandas as pd
import re
import json
from tqdm import tqdm
import torch
import gc
from util import df, df2, calculate_accuracy
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from extraction import output_json
from summary import extracted_summaries
torch.random.manual_seed(0)

tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/phi-3/pytorch/phi-3.5-mini-instruct/2", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
"/kaggle/input/phi-3/pytorch/phi-3.5-mini-instruct/2",
device_map="cuda",
torch_dtype="auto",
trust_remote_code=True,
)



accuracy = calculate_accuracy(output_json, df2)
print(f"Accuracy: {accuracy:.2f}%")

print(extracted_summaries)

