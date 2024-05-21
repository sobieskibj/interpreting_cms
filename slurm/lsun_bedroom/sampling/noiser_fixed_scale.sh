#!/usr/bin/env bash
#SBATCH --partition short
#SBATCH --account=mi2lab-hi
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time 2:00:00
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

T_STEPS=($(seq -s ' ' 10 10 150))

N_COMBS=${#T_STEPS[@]}

for ITER_ID in $(seq 0 $(($N_COMBS - 1))); do

    PARAMS=($(python src/utils/scripts/get_id_from_cart_product.py \
        -l "${T_STEPS[@]}" \
        --id $ITER_ID | tr -d '[],'))

    T_STEP=${PARAMS[0]}

    srun python src/main.py \
    --config-name cm_sampling_noiser_fixed_scale_lsun_bedroom \
    sampler.t_step=$T_STEP \
    sampler.k=50

done