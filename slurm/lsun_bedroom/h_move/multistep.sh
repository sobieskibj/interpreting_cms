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
conda activate icm

# run exp
cd /home2/faculty/bsobieski/icm

# get script parameters
WANDB_MODE=online

export HYDRA_FULL_ERROR=1

declare -A ID_TO_SHAPE

ID_TO_SHAPE=(
["input_0"]="[256, 256, 256]" \
["input_1"]="[256, 256, 256]" \
["input_2"]="[256, 256, 256]" \
["input_3"]="[256, 128, 128]" \
["input_4"]="[256, 128, 128]" \
["input_5"]="[256, 128, 128]" \
["input_6"]="[256, 64, 64]" \
["input_7"]="[512, 64, 64]" \
["input_8"]="[512, 64, 64]" \
["input_9"]="[512, 32, 32]" \
["input_10"]="[512, 32, 32]" \
["input_11"]="[512, 32, 32]" \
["input_12"]="[512, 16, 16]" \
["input_13"]="[1024, 16, 16]" \
["input_14"]="[1024, 16, 16]" \
["input_15"]="[1024, 8, 8]" \
["input_16"]="[1024, 8, 8]" \
["input_17"]="[1024, 8, 8]" \
["middle_0"]="[1024, 8, 8]" \
["output_0"]="[1024, 8, 8]" \
["output_1"]="[1024, 8, 8]" \
["output_2"]="[1024, 16, 16]" \
["output_3"]="[1024, 16, 16]" \
["output_4"]="[1024, 16, 16]" \
["output_5"]="[1024, 32, 32]" \
["output_6"]="[512, 32, 32]" \
["output_7"]="[512, 32, 32]" \
["output_8"]="[512, 64, 64]" \
["output_9"]="[512, 64, 64]" \
["output_10"]="[512, 64, 64]" \
["output_11"]="[512, 128, 128]" \
["output_12"]="[256, 128, 128]" \
["output_13"]="[256, 128, 128]" \
["output_14"]="[256, 256, 256]" \
["output_15"]="[256, 256, 256]" \
["output_16"]="[256, 256, 256]" \
["output_17"]="[256, 256, 256]" \
)

TIMESTEPS_LIST=(
    "[0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150]"
)

for TIMESTEPS in "${TIMESTEPS_LIST[@]}"; do
    for IDX in "${!ID_TO_SHAPE[@]}"; do
        srun python src/main.py \
        --config-name cm_h_move_multistep_lsun_bedroom \
        asset.h_move.norm_scale=1.0 \
        asset.h_move.id=$IDX \
        "asset.h_move.shape=${ID_TO_SHAPE[$IDX]}" \
        "sampler.ts=$TIMESTEPS"
    done
done
