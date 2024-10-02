import argparse
from datasets import load_dataset, concatenate_datasets, DatasetDict
from huggingface_hub import HfApi, HfFolder

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Merge datasets and push to Hugging Face Hub")
parser.add_argument('--hf_id', type=str, required=True, help='Hugging Face ID')
parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon value')
parser.add_argument('--rho', type=float, default=1, help='Rho value')
parser.add_argument('--phi', type=float, default=1, help='Phi value')
args = parser.parse_args()

# Extract the command-line arguments
hf_id = args.hf_id
epsilon = args.epsilon
rho = args.rho
phi = args.phi

# Define dataset paths with descriptive names
dataset_paths = {
    "safety_base": f"{hf_id}/hh-rlhf-safety-sampled-5000-train-500-test",
    "filtered_split_1": f"{hf_id}/PKU-SafeRLHF-Filtered-epsilon-{epsilon}-Split-1-rho-{rho}-phi-{phi}",
    "filtered_split_2": f"{hf_id}/PKU-SafeRLHF-Filtered-epsilon-{epsilon}-Split-2-rho-{rho}-phi-{phi}",
    "filtered_split_3": f"{hf_id}/PKU-SafeRLHF-Filtered-epsilon-{epsilon}-Split-3-rho-{rho}-phi-{phi}",
    "filtered_split_4": f"{hf_id}/PKU-SafeRLHF-Filtered-epsilon-{epsilon}-Split-4-rho-{rho}-phi-{phi}",
    "filtered_split_5": f"{hf_id}/PKU-SafeRLHF-Filtered-epsilon-{epsilon}-Split-5-rho-{rho}-phi-{phi}",
}

# Load all datasets
datasets = {name: load_dataset(path) for name, path in dataset_paths.items()}

# Define a function to merge datasets
def merge_datasets(*datasets):
    return concatenate_datasets(datasets)

# Define a function to save and push datasets
def save_and_push(dataset, name):
    dataset.save_to_disk(f"{name}")
    
    # Initialize the Hugging Face API
    api = HfApi()
    
    # Push the dataset to Hugging Face
    dataset.push_to_hub(name)

# Merge datasets for train and test splits
train_splits = {name: datasets[name]['train'] for name in datasets}
test_splits = {name: datasets[name]['test'] for name in datasets}

# Perform merging
def merge_and_push_datasets(train_splits, test_splits, dataset_names):
    train_datasets = [train_splits[name] for name in dataset_names]
    test_datasets = [test_splits[name] for name in dataset_names]
    
    merged_train = merge_datasets(*train_datasets)
    merged_test = merge_datasets(*test_datasets)
    
    merged_dataset = DatasetDict({
        'train': merged_train,
        'test': merged_test
    })
    
    # Create the repository name using the dataset combination
    dataset_name = "-".join(dataset_names)
    repo_name = f"{hf_id}/{dataset_name}"
    
    save_and_push(merged_dataset, repo_name)

# Perform merging for each combination and push to Hugging Face
dataset_combinations = [
    ['safety_base', 'filtered_split_1'],
    ['safety_base', 'filtered_split_1', 'filtered_split_2'],
    ['safety_base', 'filtered_split_1', 'filtered_split_2', 'filtered_split_3'],
    ['safety_base', 'filtered_split_1', 'filtered_split_2', 'filtered_split_3', 'filtered_split_4'],
    ['safety_base', 'filtered_split_1', 'filtered_split_2', 'filtered_split_3', 'filtered_split_4', 'filtered_split_5']
]

for combination in dataset_combinations:
    merge_and_push_datasets(train_splits, test_splits, combination)
