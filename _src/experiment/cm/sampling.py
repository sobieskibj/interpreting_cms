import torch
from omegaconf import DictConfig
from hydra.utils import instantiate

import utils
from .sampling_utils import append_zero

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
    return model, diffusion, sampler

def get_dataloader(config, fabric):
    return fabric.setup_dataloaders(instantiate(config.dataset))

def get_class_cond_bool(config):
    return config.model.num_classes is not None

def get_sampler_name(sampler_target):
    return sampler_target.split('.')[-1]

def get_sigmas_karras(config):
    """Constructs the noise schedule of Karras et al. (2022)."""
    steps = config.exp.steps
    sampler_name = get_sampler_name(config.sampler._target_)
    n = steps + 1 if sampler_name == 'progdist' else steps
    ramp = torch.linspace(0, 1, n)
    rho = config.exp.rho
    min_inv_rho = config.diffusion.sigma_min ** (1 / rho)
    max_inv_rho = config.diffusion.sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas)

def get_denoiser(config, model, diffusion, is_class_cond):

    clip_denoised = config.exp.clip_denoised
    model_kwargs = {}

    if is_class_cond:
        model_kwargs['y'] = torch.randint(
            low = 0, 
            high = config.model.num_classes, 
            size = (config.exp.batch_size,))

    def denoiser(x_t, sigma):
        _, denoised = diffusion.denoise(model, x_t, sigma, **model_kwargs)
        if clip_denoised:
            denoised = denoised.clamp(-1, 1)
        return denoised
    
    return denoiser

def get_sigma_max(config):
    return config.diffusion.sigma_max

def run(config: DictConfig):
    utils.preprocess_config(config)
    utils.setup_wandb(config)

    fabric = get_fabric(config)

    model, diffusion, sampler = get_components(config, fabric)

    dataloader = get_dataloader(config, fabric)

    with fabric.init_tensor():
        
        is_class_cond = get_class_cond_bool(config)
        sigmas = get_sigmas_karras(config)
        sigma_max = get_sigma_max(config)

        for batch_idx, batch_noise in enumerate(dataloader):
            log.info(f'Batch index: {batch_idx}')
            batch_x_T = batch_noise * sigma_max

            denoiser = get_denoiser(config, model, diffusion, is_class_cond)

            batch_x_0 = sampler(denoiser, batch_x_T, sigmas)
            batch_x_0 = batch_x_0.clamp(-1, 1)
    