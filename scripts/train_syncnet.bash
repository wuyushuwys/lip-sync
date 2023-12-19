#!/bin/bash

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

python -m torch.distributed.run --nproc_per_node $num_gpus --master_port $MASTER_PORT train_syncnet.py \
  --config ${model}.yml \
  --dataset ${dataset} \
  --model ${model} \
  --job_dir "${job_dir}" \
  --batch_size 64 \
  --epochs 100
