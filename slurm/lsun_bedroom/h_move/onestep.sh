#!/usr/bin/env bash
#SBATCH --partition short
#SBATCH --account=mi2lab-normal
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time 10:00:00
#SBATCH --job-name=icm
#SBATCH --output=slurm_logs/icm-%A.log

# echo file content to logs
script_path=$(readlink -f "$0")
cat $script_path

# load slurm
source /etc/profile.d/slurm.sh

# activate env
source /raid/shared/$USER/conda/etc/profile.d/conda.sh
conda activate inp_exp

# run exp
cd /home2/faculty/bsobieski/inp_exp

# get script parameters
WANDB_MODE=online

export HYDRA_FULL_ERROR=1

PATH_H_MOVE_CKPT=outputs/2024-04-08/16-36-49/h_move.pt

srun python src/main.py \
--config-name cm_h_move_onestep_lsun_bedroom \
h_move.path_load=$PATH_H_MOVE_CKPT