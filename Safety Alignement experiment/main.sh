pip install datasets
pip install huggingface_hub


YOUR_HF_ID='....'

#Prepare real data
python prepareRealData.py --hf_id $YOUR_HF_ID

#Split synthetic data

python splitSyntheticData.py --hf_id $YOUR_HF_ID

# Sweeping with epsilon 

epsilon=0.1
python prepareWithEpsilon.py --hf_id $YOUR_HF_ID --epsilon $epsilon

# Apply rho and phi filter

rho=0.5
phi=0.5
python push_dataset.py --hf_id $YOUR_HF_ID --epsilon $epsilon --rho $rho --phi $phi

# Prepare final data

python prepareFinalData.py --hf_id $YOUR_HF_ID --epsilon $epsilon --rho $rho --phi $phi


# Dowload alignment Handbook

git clone https://github.com/huggingface/alignment-handbook.git
cd alignment-handbook

sed -i '134s/python_requires=">=3.10.9"/python_requires=">=3.10.8"/' setup.py && \
sed -i '69s/trl>=0.9.6/trl>=0.8.2/' setup.py


python -m pip install .

python -m pip install flash-attn --no-build-isolation

huggingface-cli login

cp scripts/run_dpo.py scripts/run_ipo.py && \
sed -i '203s/loss_type=training_args.loss_type/loss_type="ipo"/' scripts/run_ipo.py


# Create the yaml files for the trainings.

mkdir -p recipes/safeFalcon11Binstruct/ipo


bash createyaml_file_step0.sh

bash create_yaml_file_step1_5.sh


#Launch the finetuning via IPO

for i in {0..5}; do
  echo "Launching step${i}..."
  ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml scripts/run_ipo.py recipes/safeFalcon11Binstruct/ipo/config_qlora_step${i}.yaml
done



# Push the Lora Adaptors to Hugging Face. 

bash createPush.sh


#!/bin/bash

# Check if the Hugging Face token file exists
if [ -f ~/.huggingface/token ]; then
    # Retrieve the Hugging Face token
    hf_token=$(cat ~/.huggingface/token)
    echo "Your Hugging Face token is: $hf_token"
else
    echo "Hugging Face token not found. Please log in using 'huggingface-cli login'."
fi


for step in {0..5}; do
  # Define the repository name and folder path
  repo_name="Safe_falcon-11b-synthetic_rho${rho}_phi${phi}_step${step}"
  folder_path="data/Safe_falcon-11b-synthetic_rho${rho}_phi${phi}_step${step}"
  
  # Run the Python script with the arguments
  python data/upload_to_hf.py --repo_name $repo_name --folder_path $folder_path --user_or_org $YOUR_HF_ID --hf_token $hf_token
  
  echo "Uploaded $repo_name to Hugging Face"
done



cd ..

pip install torch
pip install accelerate
pip install transformers
pip install peft


# Generate the responses for the safety evalution


excel_file="prompt_alert.csv"


for step in {0..5}; do

    repo_name="Safe_falcon-11b-synthetic_rho${rho}_phi${phi}_step${step}"
    safety_layer="${YOUR_HF_ID}/${repo_name}"
    output_file="Safe_falcon-11b-synthetic_rho${rho}_phi${phi}_step${step}-alert.jsonl"

    python generateResponses.py --temperature 1.0 --safety_layer $safety_layer --input_file $excel_file --output_file $output_file
done


#Merge files before safety evaluation


#!/bin/bash

# Set the input parameters
epsilon=0.5
rho=0.2
phi=0.9

# Run the Python script for steps 1 to 5
for step in {0..5}; do
    python merge_responses.py \
        --original_file alert.jsonl \
        --response_file Safe_falcon-11b-synthetic_epsilon${epsilon}_rho${rho}_phi${phi}_step${step}-alert.jsonl \
        --output_file output_Safe_falcon-11b-synthetic_epsilon${epsilon}_rho${rho}_phi${phi}_step${step}-alert.jsonl
done



#Evaluate the safety using Llama Guard 3 8B

python safetyEvaluation.py --epsilon $epsilon --rho $rho --phi $phi


