# Interpreting Consistency Models

## Virtual environment

To create a proper virtual environment, first run

```sh
conda create -n icm python=3.10
conda activate icm
pip3 install torch torchvision torchaudio
conda install mpi4py
cd _legacy
pip3 install -e .
cd ..
pip3 install hydra-core --upgrade
pip install lightning wandb tensorboard

# Athena
conda create -n icm python=3.10
conda activate icm
module load CUDA/12.0.0
conda install mpi4py
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
export LD_LIBRARY_PATH=/net/pr2/projects/plgrid/plggtriplane/sobieskibj/miniconda3/envs/icm_v2/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
cd _legacy
pip3 install -e .
pip3 install hydra-core --upgrade
pip install lightning wandb tensorboard
```

This ensures that both the legacy CMs code and new codebase can be run with the same environment.

## Reproducing the results

To reproduce the results, simply run

```sh
export HYDRA_FULL_ERROR=1
export WANDB_MODE=online

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
```
