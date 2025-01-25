import argparse
from datasets import load_dataset, DatasetDict, Dataset
from huggingface_hub import HfApi

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Split dataset and push to Hugging Face Hub")
parser.add_argument('--hf_id', type=str, required=True, help='Hugging Face ID')
args = parser.parse_args()

# Load the dataset
dataset = load_dataset("PKU-SafeRLHF-Processed")

# Function to split dataset into 5 parts
def split_into_parts(dataset, num_parts=5):
    # Shuffle the dataset
    shuffled_dataset = dataset.shuffle(seed=42)
    
    # Calculate the size of each split
    split_size = len(shuffled_dataset) // num_parts
    
    # Create a list to store the split datasets
    split_datasets = []
    
    for i in range(num_parts):
        start_idx = i * split_size
        end_idx = start_idx + split_size if i < num_parts - 1 else len(shuffled_dataset)
        
        split_datasets.append(shuffled_dataset.select(range(start_idx, end_idx)))
    
    return split_datasets

# Split the train and test datasets into 5 parts each
train_splits = split_into_parts(dataset['train'], num_parts=5)
test_splits = split_into_parts(dataset['test'], num_parts=5)

# Optionally save the splits as separate files or datasets
for i, split in enumerate(train_splits):
    split.save_to_disk(f"train_split_{i+1}.jsonl")

for i, split in enumerate(test_splits):
    split.save_to_disk(f"test_split_{i+1}.jsonl")

# Initialize the API
api = HfApi()

# Function to push a dataset split to the Hugging Face Hub
def push_to_hub(dataset, repo_id, split_name):
    dataset.push_to_hub(repo_id, split=split_name)

# Define your repository name
repo_name = "PKU-SafeRLHF-Processed-Splits"

# Use the Hugging Face ID from the command-line argument
hf_id = args.hf_id

# Create the full repository path
repo_id = f"{hf_id}/{repo_name}"

# Push each train split to the Hugging Face Hub
for i, split in enumerate(train_splits):
    split_name = f"train_split_{i+1}"
    push_to_hub(split, repo_id, split_name)

# Push each test split to the Hugging Face Hub
for i, split in enumerate(test_splits):
    split_name = f"test_split_{i+1}"
    push_to_hub(split, repo_id, split_name)
