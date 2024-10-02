import argparse
import random
from datasets import load_dataset, DatasetDict
from huggingface_hub import HfApi, HfFolder

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Filter dataset and push to Hugging Face Hub")
parser.add_argument('--hf_id', type=str, required=True, help='Hugging Face ID')
parser.add_argument('--epsilon', type=float, required=True, help='Epsilon value')
parser.add_argument('--rho', type=float, required=True, help='Rho value for filtering')
parser.add_argument('--phi', type=float, required=True, help='Phi value for filtering')
args = parser.parse_args()

# Extract arguments
hf_id = args.hf_id
epsilon = args.epsilon
rho = args.rho
phi = args.phi

# Filtering function based on sweep status
def filter_rows(example, rho, phi):
    if example['sweep_status'] == "swapped":
        # Keep with probability rho
        return random.random() < rho
    elif example['sweep_status'] == "not_swapped":
        # Keep with probability phi
        return random.random() < phi
    return False

# Function to remove 'sweep_status' column
def remove_sweep_status(example):
    example.pop('sweep_status', None)
    return example

# Initialize the API
api = HfApi()

# List of split names using Hugging Face ID and epsilon value
split_names = [
    f"{hf_id}/PKU-SafeRLHF-Swept-epsilon-{epsilon}-Split-1",
    f"{hf_id}/PKU-SafeRLHF-Swept-epsilon-{epsilon}-Split-2",
    f"{hf_id}/PKU-SafeRLHF-Swept-epsilon-{epsilon}-Split-3",
    f"{hf_id}/PKU-SafeRLHF-Swept-epsilon-{epsilon}-Split-4",
    f"{hf_id}/PKU-SafeRLHF-Swept-epsilon-{epsilon}-Split-5"
]

# Process each split
for idx, split_name in enumerate(split_names):
    # Load the dataset split
    split_dataset = load_dataset(split_name)
    
    # Apply filtering to the train split
    filtered_train_split = split_dataset['train'].filter(lambda x: filter_rows(x, rho, phi))
    
    # Apply filtering to the test split
    filtered_test_split = split_dataset['test'].filter(lambda x: filter_rows(x, rho, phi))
    
    # Remove the 'sweep_status' column from both splits
    filtered_train_split = filtered_train_split.map(remove_sweep_status)
    filtered_test_split = filtered_test_split.map(remove_sweep_status)
    
    # Create the DatasetDict with filtered splits
    filtered_split_dataset = DatasetDict({
        'train': filtered_train_split,
        'test': filtered_test_split
    })
    
    # Define the repository name with epsilon, rho, and phi values
    repo_name = f"PKU-SafeRLHF-Filtered-epsilon-{epsilon}-Split-{idx+1}-rho-{rho}-phi-{phi}"
    
    # Use the Hugging Face ID from the command-line argument
    repo_id = f"{hf_id}/{repo_name}"
    
    # Push the filtered dataset to Hugging Face Hub
    filtered_split_dataset.push_to_hub(repo_id)
