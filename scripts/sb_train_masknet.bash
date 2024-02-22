#!/bin/bash

#SBATCH --nodes=2
#SBATCH --time=24:00:00
#SBATCH --job-name=vqgan
#SBATCH --cpus-per-task=64
#SBATCH --partition=ce-mri
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=250G
#SBATCH --output=%j.log

export NCCL_P2P_DISABLE=1 # IN AMD+A100 cluster
export MASTER_PORT=$(((RANDOM % 1000 + 5000)))
num_gpus=$(nvidia-smi --list-gpus | wc -l)

# Experiments

dataset=ffhq
model=masknet

now=$(date +'%b%d-%H')

experiment_name=$1

if [ -z $experiment_name ]; then
  job_dir=runs/train_${model}_${dataset}
else
  job_dir=runs/train_${model}_${dataset}_${experiment_name}
fi

printf '%s\n' "Training on ${num_gpus} GPU ${CUDA_VISIBLE_DEVICES}"

srun torchrun --nproc_per_node $num_gpus --master_port $MASTER_PORT train_masknet.py \
  --config masknet.yml \
  --dataset ffhq celeba hdtf_images \
  --arch vqgan \
  --job_dir "${job_dir}" \
  --batch_size 8 \
  --num_workers 8 \
  --epochs 100
