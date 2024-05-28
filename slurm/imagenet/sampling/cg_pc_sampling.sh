#!/usr/bin/env bash
#SBATCH --partition short
#SBATCH --account=mi2lab-hi
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time 12:00:00
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

for T in $(seq 5 5 200); do

    srun python src/main.py \
    --config-name cm_sampling_cg_pc_imagenet \
    "exp.ts=[0,$T]" \
    'exp.ks=[31,31]' \
    exp.batch_size=2 \
    exp.n_samples=2 \
    exp.step_size=0.01

done