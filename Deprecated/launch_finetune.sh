#!/bin/bash
#SBATCH --job-name=tuab_large_no_high
#SBATCH --nodes=1 
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32 
#SBATCH --time=16:00:00 
#SBATCH --mem=512G
#SBATCH --output=/home/hice1/mchen439/scratch/EEG-Foundation-Model/slurm_outputs/tuab_large_no_high%j.out

eval "$(conda shell.bash hook)"   
conda activate Wavelet

echo "Launching finetune..." 

python finetune_downstream.py

echo "Finetune complete."
exit 0

