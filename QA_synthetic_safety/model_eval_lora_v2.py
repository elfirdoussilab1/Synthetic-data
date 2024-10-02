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
parser.add_argument('--tok', type=int, help='type of tokenizer', default=0)
parser.add_argument('--runs', type=int, help='runs per model', default=3)

args = parser.parse_args()
model_path = args.model
model_base = args.base
n_runs = args.runs

if args.tok == 0: 
    tokenizer = AutoTokenizer.from_pretrained("philschmid/gemma-tokenizer-chatml")
else:
    tokenizer = AutoTokenizer.from_pretrained(model_base)
    
lora_path = snapshot_download(repo_id=model_path)

def gen_from_model(x):
    messages = x[:-1]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True).to("cuda")
    of = input_ids.input_ids.shape[1]
    outputs = model.generate(**input_ids, max_new_tokens=40)
    ret = tokenizer.decode(outputs[0][of:])
    return ret


def get_response(inp):
    ret = gen_from_model(inp)
    if ("UNSAFE" in ret):
        return "unsafe"
    elif "SAFE" in ret:
        return "safe"
    return "none"


test = load_dataset("ai-theory/PKU-SFT-Safety")["test"]
df_test = test.to_pandas()


model_name = model_path[model_path.find("/")+1:]
file_name = f"eval_resp_v2/{model_name}.json"

def gen_responses(row):
    row["pred"] = get_response(row["messages"])
    return row

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
    temp = 0.2
    runs = []
    llm = LLM(model=model_base, max_model_len=4096, tensor_parallel_size=4, gpu_memory_utilization=0.7, enable_lora=True, max_lora_rank=64)
    for i in range(n_runs):
        sampling_params = SamplingParams(temperature=temp, top_p=0.97, max_tokens=40)
        output = llm.generate(formatted, sampling_params, lora_request=LoRARequest("adapter", 1, lora_path))
        runs.append(output)

    df_test_out = pd.DataFrame(columns = df_test.columns)
    for (i, row), j in tqdm(zip(df_test.iterrows(),range(len(runs[0]))),total=len(runs[0]),desc="post generation processing"):
        for ri in range(len(runs)):
            run = runs[ri]
            out = run[j]
            res= classify_resp(out.outputs[0].text)
            row[f"pred_{ri+1}"] = res
        df_test_out = pd.concat([df_test_out, pd.DataFrame([row])])
    df_test_out.to_json(f"eval_resp_v2/{model_name}.json")
    print(f"response stored in eval_resp_v2/{model_name}.json")
    df_res = df_test_out

print("Metrics compute in progress")
acc = np.array([round(accuracy_score(df_test["category"],df_res[f"pred_{i+1}"]),3) for i in range(n_runs)])

avg , std = round(acc.mean(),3), round(acc.std(),3)
print(f"Results for {model_path}")
print(f"Accuracy: {acc}\navg: {avg}\nstd: {std}")