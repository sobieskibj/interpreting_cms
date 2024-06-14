#!/usr/bin/env bash
#SBATCH --partition long
#SBATCH --account=mi2lab-hi
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time 5-00:00:00
#SBATCH --job-name=icm
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --output=slurm_logs/icm-%A.log

# echo file content to logs
script_path=$(readlink -f "$0")
cat $script_path

# load slurm
source /etc/profile.d/slurm.sh

# allow worker connection
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO

# activate env
source /raid/shared/$USER/conda/etc/profile.d/conda.sh
conda activate icm

# run exp
cd /home2/faculty/bsobieski/icm

export HYDRA_FULL_ERROR=1
wandb online

PATH_CKPT=weights/celebahq/training_ckpts/openai-2024-06-10-20-18-10-701815/model037500.pt

srun python src/main.py \
--config-name cm_training_isolation_celebahq \
exp.path_checkpoint=$PATH_CKPT \
fabric.devices=8 \
fabric.num_nodes=1