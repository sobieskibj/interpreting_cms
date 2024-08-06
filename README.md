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

## Test run

After creating the virtual environment, you can perform a test run by sampling images on imagenet.

Start with downloading the checkpoint

```sh
mkdir weights
cd weights
wget https://openaipublic.blob.core.windows.net/consistency/ct_imagenet64.pt
```

To run sampling using our codebase, use

```sh
HYDRA_FULL_ERROR=1 python src/main.py --config-name cm_sampling_imagenet
```

To run sampling with old codebase, use

```sh
cd _legacy

mpiexec -n 1 python scripts/image_sample.py \
--batch_size 256 \
--training_mode consistency_distillation \
--sampler onestep \
--model_path ../weights/ct_imagenet64.pt \
--attention_resolutions 32,16,8 \
--class_cond True \
--use_scale_shift_norm True \
--dropout 0.0 \
--image_size 64 \
--num_channels 192 \
--num_head_channels 64 \
--num_res_blocks 3 \
--num_samples 500 \
--resblock_updown True \
--use_fp16 True \
--weight_schedule uniform
```

## Experiments

Ready-to-use configs for the following experiments are available:

- sampling - sample images using a pretrained CM
    - using one-step sampler
    ```sh
    python src/main.py --config-name cm_sampling_onestep_imagenet
    ```
    - using multi-step sampler
    ```sh
    python src/main.py --config-name cm_sampling_multistep_imagenet
    ```
- h-move - move the representation of a given UNet block along a random direction during sampling from a fixed noise
    - using one-step sampler
    ```sh
    python src/main.py --config-name cm_h_move_onestep_imagenet
    ```
    - using multi-step sampler
    ```sh
    python src/main.py --config-name cm_h_move_multistep_imagenet
    ```

## Tips

To avoid conflicts with packages named in the same way as dirs in root, use

```python
import importlib
hf_datasets = importlib.import_module('datasets')
```