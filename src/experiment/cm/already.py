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
    already_loss = fabric.setup(instantiate(config.already_loss))
    h_move = fabric.setup(instantiate(config.h_move))
    return model, diffusion, inverter, already_loss, h_move


def get_dataloader(config, fabric):
    return fabric.setup_dataloaders(instantiate(config.dataset))


def get_denoiser(config, model, diffusion):
    # NOTE: no class conditioning for now

    model_kwargs = {}

    def denoiser(x_t, sigma):
        _, denoised = diffusion.denoise(model, x_t, sigma, **model_kwargs)
        return denoised
    
    return denoiser


def run(config: DictConfig):
    utils.preprocess_config(config)
    utils.setup_wandb(config)

    # make fabric
    fabric = get_fabric(config)

    # setup components and dataloader
    model, diffusion, inverter, already_loss = get_components(config, fabric)
    dataloader = get_dataloader(config, fabric)

    # automatically move each created tensor to proper device
    with fabric.init_tensor():

        # create denoiser with no class conditioning
        denoiser = get_denoiser(config, model, diffusion)

        for batch_idx, batch_x in enumerate(dataloader):

            # find noise that inverts the original image
            batch_noise = inverter.invert(batch_x, denoiser)

            # optimize h move wrt to already loss
            