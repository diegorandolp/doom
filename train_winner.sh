#!/bin/bash
#SBATCH -A investigacion2
#SBATCH --job-name=salmas
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
  --num_workers=40 \
  --num_envs_per_worker=8 \
  --batch_size=2048 \
  --rnn_size=1024 \
  --use_rnn=True \
  --rnn_type=gru \
  --rollout=256 \
  --recurrence=256 \
  --normalize_input=True \
  --normalize_returns=True \
  --num_epochs=2 \
  --gamma=0.997 \
  --exploration_loss_coeff=0.002 \
  --learning_rate=0.0001 \
  --max_grad_norm=4.0 \
  --env_frameskip=2 \
  --env_framestack=1 \
  --with_wandb=False \
  --save_every_sec=3600 \
  --keep_checkpoints=10 \
  --save_best_every_sec=300 \
  --train_for_env_steps=2000000000
