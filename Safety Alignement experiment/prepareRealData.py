import argparse
from datasets import load_dataset, DatasetDict
import random
from huggingface_hub import HfApi

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Push dataset to Hugging Face Hub")
parser.add_argument('--hf_id', type=str, required=True, help='Hugging Face ID')
args = parser.parse_args()

# Load the dataset from Hugging Face
dataset = load_dataset("yimingzhang/hh-rlhf-safety")

def process_example(example):
    # Extract the prompt from the chosen-chat
    prompt = None
    for entry in example['chosen-chat']:
        if entry['role'] == 'user':
            prompt = entry['content']
            break
    
    # Return the processed example with renamed columns and the prompt
    return {
        'prompt': prompt,
        'chosen': example['chosen-chat'],
        'rejected': example['rejected-chat'],
        'chosen-safety': example['chosen-safety']
    }

# Apply the processing function to both train and test splits
processed_train = dataset['train'].map(process_example)
processed_test = dataset['test'].map(process_example)

# Filter based on 'chosen-safety' being 'safe'
filtered_train = processed_train.filter(lambda example: example['chosen-safety'] == 'safe')
filtered_test = processed_test.filter(lambda example: example['chosen-safety'] == 'safe')

# Keep only the required columns
filtered_train = filtered_train.remove_columns([col for col in filtered_train.column_names if col not in ['prompt', 'chosen', 'rejected']])
filtered_test = filtered_test.remove_columns([col for col in filtered_test.column_names if col not in ['prompt', 'chosen', 'rejected']])

# Shuffle and sample 5000 rows from the filtered training split
train_sampled = filtered_train.shuffle(seed=42).select(range(5000))

# Shuffle and sample 500 rows from the filtered testing split
test_sampled = filtered_test.shuffle(seed=42).select(range(500))

# Create a DatasetDict with the sampled train and test splits
split_datasets = DatasetDict({
    'train': train_sampled,
    'test': test_sampled
})

# Initialize the API
api = HfApi()

# Use the Hugging Face ID from the command-line argument
hf_id = args.hf_id

# Define your repository name
repo_name = "hh-rlhf-safety-sampled-5000-train-500-test"
repo_id = f"{hf_id}/{repo_name}"

# Push the split datasets to Hugging Face Hub
split_datasets.push_to_hub(repo_id)
