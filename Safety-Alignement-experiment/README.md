
# Safe Falcon-11B Synthetic Data Training and Evaluation Pipeline

This repository contains a full pipeline to prepare, fine-tune, and evaluate a Safe Falcon-11B model using synthetic data. The steps include data preparation, sweeping with epsilon, applying filters, and pushing the model to Hugging Face. Additionally, safety evaluations are conducted using Llama Guard 3 8B.

## Installation

First, install the necessary dependencies:

```bash
pip install datasets
pip install huggingface_hub
pip install torch
pip install accelerate
pip install transformers
pip install peft
pip install jsonlines
pip install flash-attn --no-build-isolation
```

## Hugging Face Token

You need to be logged into the Hugging Face CLI to push the datasets and models to your Hugging Face account.

```bash
huggingface-cli login
```

Once logged in, you can retrieve the token using:

```bash
hf_token=$(cat ~/.huggingface/token)
```

## Pipeline Steps

### 1. Prepare Real Data

```bash
YOUR_HF_ID='your_huggingface_id'
python prepareRealData.py --hf_id $YOUR_HF_ID
```

### 2. Split Synthetic Data

```bash
python splitSyntheticData.py --hf_id $YOUR_HF_ID
```

### 3. Sweep with Epsilon

```bash
epsilon=0.1
python prepareWithEpsilon.py --hf_id $YOUR_HF_ID --epsilon $epsilon
```

### 4. Apply Rho and Phi Filters

```bash
rho=0.5
phi=0.5
python push_dataset.py --hf_id $YOUR_HF_ID --epsilon $epsilon --rho $rho --phi $phi
```

### 5. Prepare Final Data

```bash
python prepareFinalData.py --hf_id $YOUR_HF_ID --epsilon $epsilon --rho $rho --phi $phi
```

### 6. Clone the Alignment Handbook

```bash
git clone https://github.com/huggingface/alignment-handbook.git
cd alignment-handbook

# Modify setup.py to adjust Python and dependency versions
sed -i '134s/python_requires=">=3.10.9"/python_requires=">=3.10.8"/' setup.py && \
sed -i '69s/trl>=0.9.6/trl>=0.8.2/' setup.py

python -m pip install .
```

### 7. Finetuning IPO Model

Copy and edit the DPO training script:

```bash
cp scripts/run_dpo.py scripts/run_ipo.py
sed -i '203s/loss_type=training_args.loss_type/loss_type="ipo"/' scripts/run_ipo.py
```

### 8. Create YAML Files for Training

Create the required YAML files for the training configurations:

```bash
mkdir -p recipes/safeFalcon11Binstruct/ipo

# Create the YAML files for the training steps
bash createyaml_file_step0.sh
bash create_yaml_file_step1_5.sh
```

### 9. Launch Fine-tuning via IPO

Run the fine-tuning steps using the `accelerate` library:

```bash
for i in {0..5}; do
  echo "Launching step${i}..."
  ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml scripts/run_ipo.py recipes/safeFalcon11Binstruct/ipo/config_qlora_step${i}.yaml
done
```

### 10. Push Lora Adaptors to Hugging Face

Run the script to push the fine-tuned model and LoRA adaptors to Hugging Face:

```bash
bash createPush.sh
```

### 11. Upload Dataset to Hugging Face

```bash
for step in {0..5}; do
  repo_name="Safe_falcon-11b-synthetic_rho${rho}_phi${phi}_step${step}"
  folder_path="data/Safe_falcon-11b-synthetic_rho${rho}_phi${phi}_step${step}"
  
  python data/upload_to_hf.py --repo_name $repo_name --folder_path $folder_path --user_or_org $YOUR_HF_ID --hf_token $hf_token
done
```

### 12. Generate Responses for Safety Evaluation

```bash
excel_file="prompt_alert.csv"

for step in {0..5}; do
  repo_name="Safe_falcon-11b-synthetic_rho${rho}_phi${phi}_step${step}"
  safety_layer="${YOUR_HF_ID}/${repo_name}"
  output_file="Safe_falcon-11b-synthetic_rho${rho}_phi${phi}_step${step}-alert.jsonl"
  
  python generateResponses.py --temperature 1.0 --safety_layer $safety_layer --input_file $excel_file --output_file $output_file
done
```

### 13. Merge Responses for Safety Evaluation

```bash
#!/bin/bash

epsilon=0.5
rho=0.2
phi=0.9

for step in {0..5}; do
    python merge_responses.py \
        --original_file alert.jsonl \
        --response_file Safe_falcon-11b-synthetic_epsilon${epsilon}_rho${rho}_phi${phi}_step${step}-alert.jsonl \
        --output_file output_Safe_falcon-11b-synthetic_epsilon${epsilon}_rho${rho}_phi${phi}_step${step}-alert.jsonl
done
```

### 14. Evaluate Safety Using Llama Guard 3 8B

Evaluate the safety of the responses using the Llama Guard 3 8B model:

```bash
python safetyEvaluation.py --epsilon $epsilon --rho $rho --phi $phi
```

---

This pipeline allows for efficient preparation, fine-tuning, and safety evaluation of synthetic data using Safe Falcon-11B. Follow each step carefully, and don't forget to customize your Hugging Face ID and tokens where needed.
