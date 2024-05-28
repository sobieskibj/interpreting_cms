#!/usr/bin/env bash
#SBATCH --partition short
#SBATCH --account=mi2lab-hi
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time 24:00:00
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
wandb online
export HYDRA_FULL_ERROR=1

STEPS_MAX=($(seq -s ' ' 10 10 200))
STEPS_EVAL=(8 22 36 50 64 78 92 106 120 134 148 162 176 190 195 200)

for STEP_MAX in ${STEPS_MAX[@]}; do

    srun python src/main.py \
    --config-name cm_inversion_onestep_imagenet \
    exp.t=$STEP_MAX \
    exp.eval_ts=\[$(IFS=, ; echo "${STEPS_EVAL[*]}")\]

done