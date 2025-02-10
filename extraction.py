from util import batchify, extract_case_summary_batch
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
        "max_new_tokens": 100,  
        "return_full_text": False,
        "temperature": 0.3,  
        "do_sample": True, 
        "top_p": 0.9  
    }
batch_size = 1
extracted_summaries = []
for batch in tqdm(batchify(df3["case_text"].tolist(), batch_size)):
    batch_summaries = extract_case_summary_batch(batch, pipe, generation_args)
    for summary in batch_summaries:
        print(summary)
        extracted_summaries.append(summary)
    torch.cuda.empty_cache()
    gc.collect()

extracted_summaries
