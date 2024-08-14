import wandb
import torch
from tqdm import tqdm
from omegaconf import DictConfig
from hydra.utils import instantiate
from torchvision.utils import make_grid

import utils

import logging
log = logging.getLogger(__name__)


def get_fabric(config):
    fabric = instantiate(config.fabric)
    fabric.seed_everything(config.exp.seed)
    fabric.launch()
    return fabric


def get_components(config, fabric):
    model = fabric.setup(instantiate(config.model))
    diffusion = instantiate(config.diffusion)
    inverter = fabric.setup(instantiate(config.asset))
    return model, diffusion, inverter


def get_dataloader(config, fabric):
    return fabric.setup_dataloaders(instantiate(config.dataset))


def get_denoiser(config, model, diffusion):
    # NOTE: no class conditioning for now

    model_kwargs = {}

    def denoiser(x_t, sigma):
        _, denoised = diffusion.denoise(model, x_t, sigma, **model_kwargs)
        return denoised
    
    return denoiser


def update_pbar(config, step, scalars):
    if step % config.exp.log_every_n == 0:
        wandb.log({f'losses/{k}': v for k, v in scalars.items()})
            # 'losses/reconstruction': loss,
            # 'losses/eval_reconstruction': eval_loss})


def log_every_n(every_n, step, x, name):
    if step % every_n == 0:
        utils.log_img(x, name)


def run(config: DictConfig):
    utils.preprocess_config(config)
    utils.setup_wandb(config)

    # make fabric
    fabric = get_fabric(config)

    # setup components and dataloader
    model, diffusion, inverter = get_components(config, fabric)
    dataloader = get_dataloader(config, fabric)