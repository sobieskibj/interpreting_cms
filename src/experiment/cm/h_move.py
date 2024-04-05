import wandb
import torch
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
    sampler = instantiate(config.sampler)
    h_move = fabric.setup(instantiate(config.asset.h_move))
    return model, diffusion, sampler, h_move

def get_dataloader(config, fabric):
    return fabric.setup_dataloaders(instantiate(config.dataset))

def get_denoiser(config, model, diffusion, h_move):
    '''
    Make denoiser function.
    '''
    model_kwargs = {'h_move': h_move}

    if utils.is_class_cond(config):
        
        if config.exp.class_id is not None:
            classes = torch.ones(config.exp.batch_size) * config.exp.class_id
            classes = classes.int()
        
        else:
            classes = torch.randint(
                low = 0, 
                high = config.model.num_classes, 
                size = (config.exp.batch_size,))
            
        model_kwargs['y'] = classes

    def denoiser(x_t, sigma):
        _, denoised = diffusion.denoise(model, x_t, sigma, **model_kwargs)
        return denoised
    
    return denoiser

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

    dataloader = get_dataloader(config, fabric)

    # automatically move each created tensor to proper device
    with fabric.init_tensor():
        
        # get denoiser and sigmas for sampling
        denoiser = get_denoiser(config, model, diffusion, h_move)
        sigmas = utils.get_sigmas_karras(config)
        sigma_max = utils.get_sigma_max(config)

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