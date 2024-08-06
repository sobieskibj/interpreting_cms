#!/usr/bin/env bash
#SBATCH --partition short
#SBATCH --account=mi2lab-normal
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time 24:00:00
#SBATCH --job-name=fid_grid
#SBATCH --output=fid_grid.log

# load slurm
source /etc/profile.d/slurm.sh

# activate env
source /raid/shared/$USER/conda/etc/profile.d/conda.sh
conda activate icm

# run exp
cd /home2/faculty/jmiksa/icm

# get script parameters
wandb online
export HYDRA_FULL_ERROR=1

declare -a samplers=("cm/imagenet/onestep" 
    "cm/imagenet/multistep" 
    "cm/imagenet/classic_pc"
    "cm/imagenet/cf" 
    "cm/imagenet/cg_pc"  
    "cm/imagenet/fixed_noise_scale" 
    "cm/imagenet/corrector_training" 
    "cm/imagenet/max_noise" 
    "cm/imagenet/multistep_fixed_noise" 
    "cm/imagenet/noiser_fixed_scale"  
    "cm/imagenet/per_step_pc" 
    "cm/imagenet/true_cg")

for sampler in "${samplers[@]}"; do

    srun python src/main.py \
    --config-name cm_fid_any_imagenet \
    'sampler=${sampler}'

done
