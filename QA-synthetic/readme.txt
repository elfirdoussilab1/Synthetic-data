step to replicate the experiment:

1- gen_synthetic_QA-self.ipynb    to generate the synthetic QA dataset
2- gen_sft_syn-qa                 to create the HF dataset for synthetic QA dataset where specific model will assigned for labelling
3- gen_synthetic_qwen.py and gen_synthetic_mistral    to label the samples
4- combine_syn_llm.ipynb          will combine the results of labelling for both qwen and mistral into single dataset.
5- vllm_llama_guard.ipynb         llama_gaurd is used to verify synthetic samples
6- gen_sft_syn_full.ipynb         will construct a dataset using phi and psi paramters generating the final synthetic dataset which can used in fine-tuning experiments


model_eval_lora.py script used to evaluate the fine-tuned models based on accuracy