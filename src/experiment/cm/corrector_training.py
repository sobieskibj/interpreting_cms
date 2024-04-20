import wandb
import torch
from omegaconf import DictConfig
from hydra.utils import instantiate

import utils

import logging
log = logging.getLogger(__name__)

def get_fabric(config):
    fabric = instantiate(config.fabric)
    fabric.seed_everything(config.exp.seed)
    fabric.launch()
    return fabric

def get_components(config, fabric):
    diffusion = instantiate(config.diffusion)
    cm = fabric.setup(instantiate(config.model))
    corrector = fabric.setup(instantiate(config.corrector))
    return corrector, cm, diffusion

def get_dataloader(config, fabric):
    return fabric.setup_dataloaders(instantiate(config.dataset))

def get_denoiser(config, model, diffusion, is_class_cond):
    '''
    Make denoiser object that includes randomized class
    conditioning signal.
    '''
    model_kwargs = {}
    # NOTE: we assume no class conditioning for now
    def denoiser(x_t, sigma):
        _, denoised = diffusion.denoise(model, x_t, sigma, **model_kwargs)
        return denoised
    
    return denoiser

def run(config: DictConfig):
    utils.preprocess_config(config)
    utils.setup_wandb(config)

    # make fabric
    fabric = get_fabric(config)

    # setup components, optimizer and dataloader
    corrector, cm, diffusion = get_components(config, fabric)
    dataloader = get_dataloader(config, fabric)

    # automatically move each created tensor to proper device
    with fabric.init_tensor():
        
        # get sigmas for sampling
        sigmas = utils.get_sigmas_karras(config)
        sigma_max = utils.get_sigma_max(config)

        for batch_idx, batch_noise in enumerate(dataloader):
            log.info(f'Batch index: {batch_idx}')
            # scale standard gaussian noise with max sigma
            batch_x_T = batch_noise * sigma_max

            # sample random trajectory

            # sample random step from the trajectory

            # run through cm

            # enhance with the corrector

            # regress wrt to gt image
