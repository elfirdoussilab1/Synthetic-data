# Dataset combinations
dataset_combinations=(
    "safety_base-filtered_split_1"
    "safety_base-filtered_split_1-filtered_split_2"
    "safety_base-filtered_split_1-filtered_split_2-filtered_split_3"
    "safety_base-filtered_split_1-filtered_split_2-filtered_split_3-filtered_split_4"
    "safety_base-filtered_split_1-filtered_split_2-filtered_split_3-filtered_split_4-filtered_split_5"
)

# Loop through the combinations and create the YAML files
for i in {1..5}; do
  combination=$(echo ${dataset_combinations[$((i-1))]} | sed 's/ /, /g')
  
  cat <<EOL > recipes/safeFalcon11Binstruct/ipo/config_qlora_step${i}.yaml
# Model arguments
model_name_or_path: tiiuae/falcon-11b-instruct     # LLM to take as input
torch_dtype: bfloat16
use_flash_attention_2: true

# LoRA arguments
use_peft: true
load_in_4bit: true
lora_r: 128
lora_alpha: 128
lora_dropout: 0.05
lora_target_modules:
- query_key_value
- dense
- dense_h_to_4h
- dense_4h_to_h

# Data training arguments

dataset_mixer:
  $hf_id/$(echo ${combination}): 1.0

dataset_splits:
#- train_prefs
 - train
 - test
#- test_prefs
preprocessing_num_workers: 12

# DPOTrainer arguments
bf16: true
beta: 0.01
do_eval: true
evaluation_strategy: steps
eval_steps: 100
gradient_accumulation_steps: 4
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false
hub_model_id: zephyr-7b-dpo-qlora
learning_rate: 5.0e-6
log_level: info
logging_steps: 10
lr_scheduler_type: cosine
max_length: 1024
max_prompt_length: 512
num_train_epochs: 1
optim: paged_adamw_32bit
output_dir: data/Safe_falcon-11b-synthetic_epsilon${epsilon}_rho${rho}_phi${phi}_step${i} # It is handy to append \`hub_model_revision\` to keep track of your local experiments
per_device_train_batch_size: 4
per_device_eval_batch_size: 8
push_to_hub: false
save_strategy: "steps"
save_steps: 100
save_total_limit: 1
seed: 42
warmup_ratio: 0.1
EOL
done
