#!/bin/bash
#SBATCH -A investigacion2
#SBATCH --job-name=doom_winner
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1            
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --output=./train_winner.out
#SBATCH --error=./train_winner.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=diego.quispe.ap@utec.edu.pe

# Navigate to directory
cd /home/diego.quispe/deep/doom/new_src

# Load environment
source ~/.bashrc
source activate base

# --- DEBUG & VERIFICATION ---
echo "=========================================="
echo "Starting Job on Host: $(hostname)"
echo "Date: $(date)"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated Mem: $SLURM_MEM_PER_NODE"
echo "GPU Info:"
nvidia-smi
echo "=========================================="

srun python train_winner.py \
  --env=doom_multi_task \
  --experiment=doom_champion_v1 \
  --algo=APPO \
  --num_workers=50 \
  --num_envs_per_worker=8 \
  --batch_size=4096 \
  --rnn_size=1024 \
  --rnn_type=gru \
  --normalize_input=True \
  --normalize_returns=True \
  --gamma=0.999 \
  --exploration_loss_coeff=0.004 \
  --learning_rate=0.00005 \
  --max_grad_norm=4.0 \
  --env_framestack=1 \
  --with_wandb=False \
  --save_every_sec=3600 \
  --keep_checkpoints=10 \
  --save_best_every_sec=300 \
  --train_for_env_steps=2000000000
