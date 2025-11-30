#!/bin/bash

#SBATCH --job-name=doom
#SBATCH --partition=gpu
# Cantidad de CPUs cores a usar:
#SBATCH --cpus-per-task=16
#SBATCH --output=./train.out
#SBATCH --error=./train.err
# Tamaño de memoria del job:
#SBATCH --mem-per-cpu=2000mb
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=diego.quispe.ap@utec.edu.pe


cd /home/diego.quispe/deep/doom


source ~/.bashrc
source activate base


srun python src/train.py
