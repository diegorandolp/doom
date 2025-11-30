#!/bin/bash


#SBATCH --account=investigacion2
#SBATCH --job-name=gpu_poor
#SBATCH --partition=gpu
# Cantidad de CPUs cores a usar:
#SBATCH -c 1
#SBATCH --output=./testing_doom_runs.out
# Tamaño de memoria del job:
#SBATCH --mem-per-cpu=2000mb
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=diego.quispe.ap@utec.edu.pe


cd /home/diego.quispe/deep/doom/src


source ~/.bashrc
source activate base


srun python collect_data.py
