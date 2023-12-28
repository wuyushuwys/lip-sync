#!/bin/bash


#SBATCH --nodes=2
#SBATCH --time=48:00:00
#SBATCH --job-name=srgan
#SBATCH --cpus-per-task=24
#SBATCH --partition=ce-mri
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --output=%j.log

export MASTER_PORT=$(((RANDOM % 1000 + 5000)))
num_gpus=$(nvidia-smi --list-gpus | wc -l)

# Experiments

dataset=hdtf
model=syncnet

now=$(date +'%b%d-%H')

experiment_name=$1

if [ -z $experiment_name ]; then
  job_dir=runs/train_${model}_${dataset}
else
  job_dir=runs/train_${model}_${dataset}_${experiment_name}
fi

printf '%s\n' "Training on ${num_gpus} GPU ${CUDA_VISIBLE_DEVICES}"

torchrun --nproc_per_node $num_gpus --master_port $MASTER_PORT train_syncnet.py \
  --config ${model}.yml \
  --dataset ${dataset} \
  --model ${model} \
  --job_dir "${job_dir}" \
  --batch_size 64 \
  --warmup_lr \
  --epochs 100
