#!/bin/bash
#SBATCH -A investigacion2
#SBATCH --job-name=check
#SBATCH --partition=gpu
# Cantidad de CPUs cores a usar:
#SBATCH --cpus-per-task=1
#SBATCH --output=./check.out
#SBATCH --error=./check.err
# Tamaño de memoria del job:
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=diego.quispe.ap@utec.edu.pe


cd /home/diego.quispe/deep/doom


source ~/.bashrc
source activate base

export CUDA_VISIBLE_DEVICES=0

#srun python src/train.py
srun python check_gpu.py
