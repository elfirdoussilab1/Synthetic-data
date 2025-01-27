import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sklearn.metrics import accuracy_score,f1_score,recall_score
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams
from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

parser = argparse.ArgumentParser(description="Programs that evaluates Content Classifiers")
parser.add_argument('--model', type=str, help='model to evaluate')
parser.add_argument('--base', type=str, help='model to evaluate', default="google/gemma-2-2b-it")

args = parser.parse_args()
model_path = args.model
model_base = args.base

# tokenizer = AutoTokenizer.from_pretrained("philschmid/gemma-tokenizer-chatml")
tokenizer = AutoTokenizer.from_pretrained(model_base)
lora_path = snapshot_download(repo_id=model_path)


org = "<replace with you HF org>"
test = load_dataset(f"{org}/PKU-SFT-Safety")["test"]
df_test = test.to_pandas()

model_name = model_path[model_path.find("/")+1:]
file_name = f"eval_resp/{model_name}.json"

def classify_resp(inp):
    if "UNSAFE" in inp:
        return "unsafe"
    elif "SAFE" in inp:
        return "safe"
    return "none"
try:
    df_res = pd.read_json(file_name)
except:
    formatted = [tokenizer.apply_chat_template(row["messages"][:-1], tokenize=False, add_generation_prompt=True) for i,row in df_test.iterrows()]
    print("Generation in progress")
    llm = LLM(model=model_base, max_model_len=4096, tensor_parallel_size=4, gpu_memory_utilization=0.7, enable_lora=True, max_lora_rank=64)
    sampling_params = SamplingParams(temperature=0.9, top_p=0.97, max_tokens=40)
    outputs = llm.generate(formatted, sampling_params, lora_request=LoRARequest("adapter", 1, lora_path))

    df_test_out = pd.DataFrame(columns = df_test.columns)
    for (i, row), out in tqdm(zip(df_test.iterrows(),outputs),total=len(outputs),desc="post generation processing"):
        res = classify_resp(out.outputs[0].text)
        row["pred"] = res
        df_test_out = pd.concat([df_test_out, pd.DataFrame([row])])
    df_test_out.to_json(f"eval_resp/{model_name}.json")
    print(f"response stored in eval_resp/{model_name}.json")
    df_res = df_test_out

print("Metrics compute in progress")
acc = round(accuracy_score(df_test["category"],df_res["pred"]),3)
recall = round(recall_score(df_test["category"],df_res["pred"],average='weighted'),3)
f1 = round(f1_score(df_test["category"],df_res["pred"],average='weighted'),3)
print(f"Results for {model_path}")
print(f"acc: {acc}\nrecall: {recall}\nf1: {f1}")