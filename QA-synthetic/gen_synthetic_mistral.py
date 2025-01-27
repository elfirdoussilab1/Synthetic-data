from pathlib import Path
import os
from time import time
import json
from argparse import ArgumentParser

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
import pandas as pd
import argparse


def main(args):
    model_name = 'mistralai/Mistral-Nemo-Instruct-2407'

    data = load_dataset(args.dataset)
    df_data = data["train"].to_pandas()
    ## dataset will have column called gen, which indicate that we expect this sample to be classified by mistral
    df_data = df_data[df_data["gen"]=="mistral7b"]
    prompts = df_data['messages']

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    formatted = [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in tqdm(prompts)]

    llm = LLM(model=model_name, max_model_len=4096, tensor_parallel_size=4, gpu_memory_utilization=0.7)
    sampling_params = SamplingParams(temperature=0.9, top_p=0.97, max_tokens=40)
    outputs = llm.generate(formatted, sampling_params)

    def set_resp(row, out):
        if "unsafe" in out:
            row["mistral7b-resp"] = "unsafe"
        elif "safe" in out:
            row["mistral7b-resp"] = "safe"
        else:
            row["mistral7b-resp"] = "none"
        return row


    out_df = pd.DataFrame(columns = df_data.columns)

    for (i, row), out in tqdm(zip(df_data.iterrows(),outputs),total=len(outputs),desc="post generation processing"):
        res = out.outputs[0].text.lower()
        new_row = set_resp(row,res)
        new_row["model_prompt"] = out.prompt
        out_df = pd.concat([out_df, pd.DataFrame([new_row])])

    print(f"dumping outputs")
    out_df.to_json(f"{args.out_dir}/mistral_responses.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="script that ask Mistral-Nemo-Instruct-2407 to label QA as safe/unsafe")
    parser.add_argument('--dataset', type=str, help='HF repo of the QA dataset')
    parser.add_argument('--out-dir', type=str, help='specific path to store labelled samples')
    
    args = parser.parse_args()
    main(args)