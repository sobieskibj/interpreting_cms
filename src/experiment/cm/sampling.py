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
    return model, diffusion, sampler

def get_dataloader(config, fabric):
    return fabric.setup_dataloaders(instantiate(config.dataset))

def get_denoiser(config, model, diffusion, is_class_cond):
    '''
    Make denoiser object that includes randomized class
    conditioning signal.
    '''
    model_kwargs = {}

    if is_class_cond:
        model_kwargs['y'] = torch.randint(
            low = 0, 
            high = config.model.num_classes, 
            size = (config.exp.batch_size,))

    def denoiser(x_t, sigma):
        _, denoised = diffusion.denoise(model, x_t, sigma, **model_kwargs)
        return denoised
    
    return denoiser

def get_sigma_max(config):
    return config.diffusion.sigma_max

def log_batch_imgs(batch_imgs):
    grid = make_grid(batch_imgs).permute(1, 2, 0)
    grid = wandb.Image(grid.numpy(force = True))
    wandb.log({'samples': grid})

def run(config: DictConfig):
    utils.preprocess_config(config)
    utils.setup_wandb(config)

    # make fabric
    fabric = get_fabric(config)

    # automatically move each created tensor to proper device
    with fabric.init_tensor():

        # setup components and dataloader
        model, diffusion, sampler = get_components(config, fabric)
        dataloader = get_dataloader(config, fabric)
        
        # get class conditioning and sigmas for sampling
        is_class_cond = utils.is_class_cond(config)
        sigmas = utils.get_sigmas_karras(config)

        for batch_idx, batch_input in enumerate(dataloader):
            log.info(f'Batch index: {batch_idx}')

            # create denoiser object separately for each batch
            # to randomize class conditoning
            denoiser = get_denoiser(config, model, diffusion, is_class_cond)

            log.info('Generating images')
            # run sampler with the denoiser to obtain images
            batch_x_0 = sampler(distiller = denoiser, x = batch_input, sigmas = sigmas)
            batch_x_0 = batch_x_0.clamp(-1, 1)

            log.info('Logging images')
            # log samples
            log_batch_imgs(batch_x_0)