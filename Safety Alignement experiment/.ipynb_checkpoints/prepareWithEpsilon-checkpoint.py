import argparse
import random
from datasets import load_dataset, DatasetDict

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Sweep columns and push to Hugging Face Hub")
parser.add_argument('--hf_id', type=str, required=True, help='Hugging Face ID')
parser.add_argument('--epsilon', type=float, required=True, help='Epsilon value for sweeping')
args = parser.parse_args()

# Extract arguments
hf_id = args.hf_id
epsilon = args.epsilon

# Function to sweep the chosen and rejected columns
def sweep_columns(example, epsilon):
    if random.random() < epsilon:
        # Swap chosen and rejected
        example['chosen'], example['rejected'] = example['rejected'], example['chosen']
        example['sweep_status'] = "swapped"
    else:
        example['sweep_status'] = "not_swapped"
    return example

# List of dataset names using Hugging Face ID
split_names = [
    f"{hf_id}/PKU-SafeRLHF-Processed-Splits_part_1",
    f"{hf_id}/PKU-SafeRLHF-Processed-Splits_part_2",
    f"{hf_id}/PKU-SafeRLHF-Processed-Splits_part_3",
    f"{hf_id}/PKU-SafeRLHF-Processed-Splits_part_4",
    f"{hf_id}/PKU-SafeRLHF-Processed-Splits_part_5"
]

# Process each split
for idx, split_name in enumerate(split_names):
    # Load the dataset split
    split_dataset = load_dataset(split_name)
    
    # Apply the sweep to the train split
    swept_train_split = split_dataset['train'].map(lambda example: sweep_columns(example, epsilon=epsilon))
    
    # Apply the sweep to the test split
    swept_test_split = split_dataset['test'].map(lambda example: sweep_columns(example, epsilon=epsilon))
    
    # Create a new DatasetDict with the swept splits
    swept_split_dataset = DatasetDict({
        'train': swept_train_split,
        'test': swept_test_split
    })
    
    # Define your repository name for the swept split
    repo_name = f"PKU-SafeRLHF-Swept-epsilon-{epsilon}-Split-{idx+1}"
    
    # Use the Hugging Face ID from the command-line argument
    repo_id = f"{hf_id}/{repo_name}"
    
    # Push the swept dataset to Hugging Face Hub
    swept_split_dataset.push_to_hub(repo_id)
