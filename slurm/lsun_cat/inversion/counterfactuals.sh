#!/usr/bin/env bash
#SBATCH --partition short
#SBATCH --account=mi2lab-hi
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time 24:00:00
#SBATCH --job-name=icm
#SBATCH --array=1-6
#SBATCH --output=slurm_logs/icm-%A-%a.log

# echo file content to logs
script_path=$(readlink -f "$0")
cat $script_path

# load slurm
source /etc/profile.d/slurm.sh

# activate env
cload ~/conda.tar
source /raid/shared/$USER/conda/etc/profile.d/conda.sh
conda activate icm

# run exp
cd /home2/faculty/bsobieski/icm

# get script parameters
wandb online
export HYDRA_FULL_ERROR=1

T_STEPS=($(seq -s ' ' 10 20 150))
LRS=(0.01 0.001)
ALPHAS=(${SLURM_ARRAY_TASK_ID}.0)
LOSSES=(lpips mse)

for ITER in {0..63}; do

    PARAMS=($(python src/utils/scripts/get_id_from_cart_product.py \
    -l "${T_STEPS[@]}" \
    -l "${LRS[@]}" \
    -l "${ALPHAS[@]}" \
    -l "${LOSSES[@]}" \
    --id $ITER | tr -d '[],'))

    srun python src/main.py \
    --config-name cm_inversion_onestep_cf_imagenet_cat \
    exp.t=${PARAMS[0]} \
    "exp.eval_ts=[${PARAMS[0]}]" \
    exp.lr=${PARAMS[1]} \
    exp.alpha=${PARAMS[2]} \
    similarity_loss=${PARAMS[3]}

done