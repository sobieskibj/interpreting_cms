from tqdm import tqdm
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
    fid = fabric.setup(instantiate(config.fid))
    return model, diffusion, sampler, fid

def get_dataloaders(config, fabric):
    data_loader, noise_loader = \
        fabric.setup_dataloaders(
            instantiate(config.dataset), instantiate(config.noise_dataset))
    return data_loader, noise_loader

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


def from_m1p1_to_01(x):
    return (x + 1.) / 2.


def log_batch_imgs(key, batch_imgs):
    grid = make_grid(batch_imgs).permute(1, 2, 0)
    grid = wandb.Image(grid.numpy(force = True))
    wandb.log({f"imgs/{key}": grid})


def get_total_length(dataloader):
    return len(dataloader.dataset) // dataloader.batch_size

def run(config: DictConfig):
    utils.preprocess_config(config)
    utils.setup_wandb(config)

    # make fabric
    fabric = get_fabric(config)

    # automatically move each created tensor to proper device
    with fabric.init_tensor():

        # setup components and dataloader
        model, diffusion, sampler, fid = get_components(config, fabric)
        data_loader, noise_loader = get_dataloaders(config, fabric)
        
        # get class conditioning and sigmas for sampling
        is_class_cond = utils.is_class_cond(config)
        sigmas = utils.get_sigmas_karras(config)

        # create containers for fid features and the iterator
        ftrs_true, ftrs_synt = [], []
        total_length = get_total_length(data_loader)
        iterator = tqdm(enumerate(zip(data_loader, noise_loader)), total = total_length)

        for batch_idx, (batch_data, batch_noise) in iterator:
            batch_data = from_m1p1_to_01(batch_data)

            # create denoiser object separately for each batch
            # to randomize class conditoning
            denoiser = get_denoiser(config, model, diffusion, is_class_cond)

            # run sampler with the denoiser to obtain images
            batch_x_0 = sampler(distiller = denoiser, x = batch_noise, sigmas = sigmas)
            batch_x_0 = batch_x_0.clamp(-1, 1)
            batch_x_0 = from_m1p1_to_01(batch_x_0)

            # compute FID features
            ftrs_true_ = fid.features(batch_data)
            ftrs_synt_ = fid.features(batch_x_0)

            # store features
            ftrs_true.append(ftrs_true_), ftrs_synt.append(ftrs_synt_)

            # optionally log images
            if batch_idx == 0:
                log_batch_imgs("true", batch_data)
                log_batch_imgs("synt", batch_x_0)

        ftrs_true, ftrs_synt = torch.cat(ftrs_true), torch.cat(ftrs_synt)
        v = fid(ftrs_synt, ftrs_true)
        log.info(f"FID: {v}")