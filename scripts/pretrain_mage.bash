#!/bin/bash

export MASTER_PORT=$(((RANDOM % 1000 + 5000)))
num_gpus=$(nvidia-smi --list-gpus | wc -l)

# Experiments

dataset=large_scale
model=mage

now=$(date +'%b%d-%H')

experiment_name=$1

if [ -z $experiment_name ]; then
  job_dir=runs/train_${model}_${dataset}
else
  job_dir=runs/train_${model}_${dataset}_${experiment_name}
fi

printf '%s\n' "Training on ${num_gpus} GPU ${CUDA_VISIBLE_DEVICES}"

torchrun --nproc_per_node $num_gpus --master_port $MASTER_PORT pretrain_mage.py \
  --config mage_pretrain.yml \
  --dataset ffhq celeba open_images imagenet \
  --model mage \
  --job_dir "${job_dir}" \
  --batch_size 16 \
  --num_workers 8 \
  --epochs 200
