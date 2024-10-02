#!/bin/bash

# Install necessary Python packages
pip install transformers accelerate peft



excel_file="prompt_alert.csv"

safety_layer="RedaAlami/Safe_falcon-11b-synthetic_epsilon0_5_rho0_2_phi0_9_step1"
output_file="Safe_falcon-11b-synthetic_epsilon0_5_rho0_2_phi0_9_step1-alert.jsonl"

python generateResponses.py --temperature 1.0 --safety_layer $safety_layer --input_file $excel_file --output_file $output_file


