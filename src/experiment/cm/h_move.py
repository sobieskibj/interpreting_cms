import wandb
import torch
from omegaconf import DictConfig
from hydra.utils import instantiate
from torchvision.utils import make_grid

import utils
from model.cm.src.samplers import append_zero

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
    sampler = instantiate(config.sampler)
    h_move = fabric.setup(instantiate(config.asset.h_move))
    return model, diffusion, sampler, h_move

def get_dataloader(config, fabric):
    return fabric.setup_dataloaders(instantiate(config.dataset))

def get_sampler_name(sampler_target):
    return sampler_target.split('.')[-1]

def get_sigmas_karras(config):
    '''Constructs the noise schedule of Karras et al. (2022).'''
    steps = config.exp.steps
    sampler_name = get_sampler_name(config.sampler._target_)
    n = steps + 1 if sampler_name == 'progdist' else steps
    ramp = torch.linspace(0, 1, n)
    rho = config.exp.rho
    min_inv_rho = config.diffusion.sigma_min ** (1 / rho)
    max_inv_rho = config.diffusion.sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas)

def get_denoiser(config, model, diffusion, h_move):
    '''
    Make denoiser function.
    '''
    clip_denoised = config.exp.clip_denoised
    model_kwargs = {'h_move': h_move}

    def denoiser(x_t, sigma):
        _, denoised = diffusion.denoise(model, x_t, sigma, **model_kwargs)
        if clip_denoised:
            denoised = denoised.clamp(-1, 1)
        return denoised
    
    return denoiser

def get_sigma_max(config):
    return config.diffusion.sigma_max

def log_batch_imgs(batch_imgs):
    grid = make_grid(batch_imgs).permute(1, 2, 0)
    # images are normalized by wandb.Image
    grid = wandb.Image(grid.numpy(force = True))
    wandb.log({'samples': grid})

def run(config: DictConfig):
    utils.preprocess_config(config)
    utils.setup_wandb(config)

    # make fabric
    fabric = get_fabric(config)

    # setup components and dataloader
    model, diffusion, sampler, h_move = get_components(config, fabric)
    denoiser = get_denoiser(config, model, diffusion, h_move)
    dataloader = get_dataloader(config, fabric)
    # TODO: assert that samplers are adding the same noise to each image

    # automatically move each created tensor to proper device
    with fabric.init_tensor():
        
        # get class conditioning and sigmas for sampling
        sigmas = get_sigmas_karras(config)
        sigma_max = get_sigma_max(config)

        for batch_idx, batch_noise in enumerate(dataloader):
            log.info(f'Batch index: {batch_idx}')
            # scale standard gaussian noise with max sigma
            batch_x_T = batch_noise * sigma_max

            log.info('Generating images')
            # run sampler with the denoiser to obtain images
            batch_x_0 = sampler(denoiser, batch_x_T, sigmas)
            batch_x_0 = batch_x_0.clamp(-1, 1)

            log.info('Logging images')
            # log samples
            log_batch_imgs(batch_x_0)