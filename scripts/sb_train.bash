#!/bin/bash


#SBATCH --nodes=4
#SBATCH --time=48:00:00
#SBATCH --job-name=srgan
#SBATCH --cpus-per-task=48
#SBATCH --partition=ce-mri
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=250G
#SBATCH --output=logs/%j.log

export NCCL_P2P_DISABLE=1  # IN AMD+A100 cluster
export MASTER_PORT=$(((RANDOM % 1000 + 5000)))
num_gpus=$(nvidia-smi --list-gpus | wc -l)

# Experiments

##################
#   placeholder  #
##################

now=$(date +'%b%d-%H')

experiment_name=$1

if [ -z $experiment_name ]; then
  job_dir=train_${now}
else
  job_dir=train_${experiment_name}_${now}
fi

if [ -d "runs/$job_dir" ]; then
    printf '%s\n' "Removing runs/$job_dir"
    rm -rf "runs/$job_dir"
fi


printf '%s\n' "Training on GPU ${CUDA_VISIBLE_DEVICES}"

srun python -m torch.distributed.run  --nproc_per_node $num_gpus --master_port $MASTER_PORT train.py
##################
#   placeholder  #
##################