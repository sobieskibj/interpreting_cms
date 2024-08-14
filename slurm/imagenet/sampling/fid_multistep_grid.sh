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

TS_ARRAY=(
    [0,40,80,120,160,200]
    [0,34,68,102,136,170,200]
    [0,29,58,87,116,145,174,200]
    [0,25,50,75,100,125,150,175,200]
    [0,20,40,60,80,100,120,140,160,180,200]
)

for TS in ${TS_ARRAY[@]}; do
    echo "sampler.ts=${TS}"
    srun python src/main.py \
    --config-name cm_fid_multistep_imagenet \
    exp.path_ckpt=weights/ct_imagenet64.pt \
    "sampler.ts=${TS}"
done