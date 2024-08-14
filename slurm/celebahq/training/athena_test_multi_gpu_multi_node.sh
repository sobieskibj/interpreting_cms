#!/usr/bin/env bash
#SBATCH --partition plgrid-gpu-a100
#SBATCH --account plgvdgs-gpu-a100
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time 0-00:10:00
#SBATCH --job-name=icm
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --output=slurm_logs/icm-%A.log

# echo file content to logs
script_path=$(readlink -f "$0")
cat $script_path

# activate env
source /net/pr2/projects/plgrid/plggtriplane/sobieskibj/miniconda3/etc/profile.d/conda.sh
conda activate icm_v2

# run exp
cd /net/pr2/projects/plgrid/plggtriplane/sobieskibj/repos/icm/_legacy/scripts

export SLURM_JOB_GPUS=1,1,1,1 SLURM_JOB_NUM_NODES=2

mpiexec python cm_train.py \
--training_mode consistency_training \
--target_ema_mode adaptive \
--start_ema 0.95 \
--scale_mode progressive \
--start_scales 2 \
--end_scales 150 \
--total_training_steps 1000000 \
--loss_norm lpips \
--lr_anneal_steps 0 \
--attention_resolutions 32,16,8 \
--class_cond False \
--use_scale_shift_norm False \
--dropout 0.0 \
--teacher_dropout 0.1 \
--ema_rate 0.9999,0.99994,0.9999432189950708 \
--global_batch_size 2048 \
--image_size 256 \
--lr 0.00005 \
--num_channels 256 \
--num_head_channels 64 \
--num_res_blocks 2 \
--resblock_updown True \
--schedule_sampler uniform \
--use_fp16 True \
--weight_decay 0.0 \
--weight_schedule uniform \
--data_dir ../../data/celebahq/data \
--microbatch 2