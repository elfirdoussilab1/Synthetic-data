import argparse
import json
import time

import pandas as pd
import torch
import tqdm
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from colorama import Fore, Style, init

from peft import PeftModel

# Parsing command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--safety_layer", type=str, required=True, help="Path to the input file")
parser.add_argument("--input_file", type=str, required=True, help="Path to the input file")
parser.add_argument("--output_file", type=str, required=True, help="Path to the output file")
args = parser.parse_args()

# Initialize Accelerator
accelerator = Accelerator()

# Load the base model and tokenizer
base_model_name = "tiiuae/falcon-11b-instruct"  # or any other base model
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

#model_safety = 'RedaAlami/falcon-11b-safe-ipo-qlora'
model_safety = args.safety_layer

base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(base_model, model_safety)


# Load the input file
df = pd.read_csv(args.input_file)

# Function to format the prompts using a chat template
def use_template(row):
    chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ 'User: \n' + message['content'] }}\n{% elif message['role'] == 'system' %}\n{{ 'System: ' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ 'Falcon:\n'  + message['content']}}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ 'Falcon:' }}\n{% endif %}\n{% endfor %}"
    request = tokenizer.apply_chat_template(
        [{'content': row['prompt'], 'role': 'user'}],
        chat_template=chat_template,
        tokenize=False,
        add_generation_prompt=True
    )
    return request

prompts_all = df.apply(use_template, axis=1).tolist()

# Prepare the model for distributed computing
model, tokenizer = accelerator.prepare(model, tokenizer)

# Manually split data among processes
split_size = len(prompts_all) // accelerator.num_processes
start_idx = accelerator.process_index * split_size
end_idx = start_idx + split_size if accelerator.process_index != accelerator.num_processes - 1 else len(prompts_all)
prompts_split = prompts_all[start_idx:end_idx]

# Batch and tokenize the prompts
def prepare_prompts(prompts, tokenizer, batch_size=30):
    batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]
    batches_tok = []
    tokenizer.padding_side = "left"
    for prompt_batch in batches:
        batches_tok.append(
            tokenizer(prompt_batch, return_tensors="pt", padding='longest', truncation=False, pad_to_multiple_of=8, add_special_tokens=False).to(model.device)
        )
    tokenizer.padding_side = "right"
    return batches_tok

# Synchronize GPUs and start the timer
accelerator.wait_for_everyone()
start = time.time()

# Processing and generation
results = []
prompt_batches = prepare_prompts(prompts_split, tokenizer)
for prompts_tokenized in tqdm.tqdm(prompt_batches):
    with torch.no_grad():
        outputs_tokenized = model.generate(
            **prompts_tokenized, max_new_tokens=70,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=args.temperature
        )
    outputs_cleaned = [tok_out[len(tok_in):] for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized)]
    outputs = tokenizer.batch_decode(outputs_cleaned, skip_special_tokens=True)
    results.extend(outputs)

# Gather results from all GPUs if distributed computing is used
results = accelerator.gather(results)

if accelerator.is_main_process:
    # Save the generated text to a file
    timediff = time.time() - start
    with open(args.output_file, "w") as write_file:
        json.dump(results, write_file)
    print(f"Total time: {timediff}s")
    print(f"Results saved to {args.output_file}")
