#!/usr/bin/env bash
#SBATCH --partition long
#SBATCH --account=mi2lab-hi
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time 48:00:00
#SBATCH --job-name=icm
#SBATCH --output=slurm_logs/icm-%A.log

# echo file content to logs
script_path=$(readlink -f "$0")
cat $script_path

# load slurm
source /etc/profile.d/slurm.sh

# activate env
source /raid/shared/$USER/conda/etc/profile.d/conda.sh
conda activate icm

# run exp
cd /home2/faculty/bsobieski/icm

# get script parameters
export HYDRA_FULL_ERROR=1
wandb online

srun python src/main.py \
--config-name cm_corrector_training_imagenet