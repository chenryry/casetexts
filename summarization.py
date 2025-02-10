from util import batchify, extract_case_details_batch
import numpy as np
import pandas as pd
import re
import json
from tqdm import tqdm
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
torch.random.manual_seed(0)
from main import model,tokenizer
from util import df,df2,df3


pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

batch_size = 1
extracted_data = []
for batch in tqdm(batchify(df3["case_text"].tolist(), batch_size)):
    batch_responses = extract_case_details_batch(batch, pipe, generation_args)
    for response in batch_responses:
        try:
            print(response)
            extracted = json.loads(response)
        except json.JSONDecodeError:
            continue
        extracted_data.append(extracted)
    torch.cuda.empty_cache()
    gc.collect()


output_json = json.dumps(extracted_data, indent=4)

