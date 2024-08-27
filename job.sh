#!/bin/bash
#SBATCH --job-name=synthetic-data
#SBATCH --output=synthetic-data.out
#SBATCH --error=synthetic-data.error
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --exclude=t01pdscgpu01
 
 
eval "$(/lustre1/tier2/users/aymane.elfirdoussi/miniconda3/bin/conda shell.bash hook)"
conda activate venv
 
python3 train_mnist.py