#!/bin/bash
#SBATCH --job-name=batch_job_sst
#SBATCH --partition=grete:interactive
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=./slurm_files/slurm-%x-%j.out
#SBATCH --error=./slurm_files/slurm-%x-%j.err
# Contributor email intentionally omitted.

mkdir -p slurm_files
source activate dnlp
python --version
python multitask_classifier.py --option finetune --task sst --use_gpu --local_files_only
