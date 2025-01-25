#!/bin/bash

# Create the 'data' directory if it doesn't exist
mkdir -p data

# Create the 'push.py' file with the required content
cat <<EOL > data/push.py
import argparse
from huggingface_hub import HfApi, upload_folder

# Define the argument parser
parser = argparse.ArgumentParser(description="Upload a model folder to Hugging Face Hub")

# Add arguments
parser.add_argument('--repo_name', type=str, required=True, help="The name of the repository on Hugging Face")
parser.add_argument('--folder_path', type=str, required=True, help="The local folder path containing the model")
parser.add_argument('--user_or_org', type=str, required=True, help="Your Hugging Face user or organization name")
parser.add_argument('--hf_token', type=str, required=True, help="Your Hugging Face token")

# Parse arguments
args = parser.parse_args()

# Assign parsed arguments to variables
repo_name = args.repo_name
folder_path = args.folder_path
user_or_org = args.user_or_org
hf_token = args.hf_token

# Initialize the API
api = HfApi()

# Create a new repository on Hugging Face
api.create_repo(
    repo_id=f"\{user_or_org}/\{repo_name}",
    token=hf_token,
    private=True
)

# Upload the model folder to the repository
upload_folder(
    repo_id=f"\{user_or_org}/\{repo_name}",
    folder_path=folder_path,
    path_in_repo="",  # Upload to the root of the repository
    token=hf_token
)
EOL

echo "push.py script has been created in the 'data' directory."
