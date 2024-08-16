#!/usr/bin/env bash
#SBATCH --partition short
#SBATCH --account=mi2lab-normal
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time 24:00:00
#SBATCH --job-name=icm
#SBATCH --array=0-7
#SBATCH --output=slurm_logs/icm-%A-%a.log

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
export WANDB_MODE=online
wandb online

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

TS=(5 25 45 65 85 105 125 135)
T=${TS[SLURM_ARRAY_TASK_ID]}

for IDX in "${!ID_TO_SHAPE[@]}"
do
    srun python src/main.py \
    --config-name cm_already_lsun_bedroom \
    inverter.n_iters=8192 \
    model.use_fp16=false \
    model.attention_type=null \
    h_move.id=$IDX \
    "h_move.shape=${ID_TO_SHAPE[$IDX]}" \
    inverter.t=$T
done