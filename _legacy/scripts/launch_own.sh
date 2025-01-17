#!/usr/bin/env bash

mpiexec -n 1 python scripts/image_sample.py \
--batch_size 256 \
--training_mode consistency_distillation \
--sampler onestep \
--model_path weights/ct_imagenet64.pt \
--attention_resolutions 32,16,8 \
--class_cond True \
--use_scale_shift_norm True \
--dropout 0.0 \
--image_size 64 \
--num_channels 192 \
--num_head_channels 64 \
--num_res_blocks 3 \
--num_samples 1024 \
--resblock_updown True \
--use_fp16 True \
--weight_schedule uniform

mpiexec -n 1 python cm_train.py \
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
--global_batch_size 256 \
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

mpiexec -n 2 python cm_train.py \
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
--global_batch_size 256 \
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