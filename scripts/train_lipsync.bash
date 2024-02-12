#!/bin/bash

export MASTER_PORT=$(((RANDOM % 1000 + 5000)))
num_gpus=$(nvidia-smi --list-gpus | wc -l)

# Experiments

dataset=hdtf
model=lipsync

now=$(date +'%b%d-%H')

experiment_name=$1

if [ -z $experiment_name ]; then
  job_dir=runs/train_${model}_${dataset}
else
  job_dir=runs/train_${model}_${dataset}_${experiment_name}
fi

printf '%s\n' "Training on ${num_gpus} GPU ${CUDA_VISIBLE_DEVICES}"

torchrun --nproc_per_node $num_gpus --master_port $MASTER_PORT train_lipsync.py \
  --config ${model}.yml \
  --dataset ${dataset} \
  --arch ${model} \
  --job_dir "${job_dir}" \
  --batch_size 8 \
  --num_workers 4 \
  --epochs 100
