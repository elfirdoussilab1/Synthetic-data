# Steps to Replicate the Experiment

1. **gen_synthetic_QA-self.ipynb**
   - Generates the synthetic QA dataset

2. **gen_sft_syn-qa.ipynb**
   - Creates the HuggingFace dataset for synthetic QA
   - Assigns specific model for labelling

3. **gen_synthetic_qwen.py** and **gen_synthetic_mistral**
   - Labels the samples

4. **combine_syn_llm.ipynb**
   - Combines the labeling results from both Qwen and Mistral into a single dataset

5. **vllm_llama_guard.ipynb**
   - Uses LLaMA Guard to verify synthetic samples

6. **gen_sft_syn_full.ipynb**
   - Constructs dataset using phi and psi parameters
   - Generates final synthetic dataset for fine-tuning experiments

7. **model_eval_lora.py**
   - Script for evaluating fine-tuned models based on accuracy
