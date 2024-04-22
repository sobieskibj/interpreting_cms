import wandb
import torch
import torch.nn.functional as F
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
    sampler = instantiate(config.sampler)(diffusion = diffusion)
    cm = fabric.setup(instantiate(config.model))
    corrector = instantiate(config.asset)
    optimizer = instantiate(config.optimizer)(params = corrector.parameters())
    corrector, optimizer = fabric.setup(corrector, optimizer)
    return corrector, optimizer, cm, diffusion, sampler

def get_dataloader(config, fabric):
    return fabric.setup_dataloaders(instantiate(config.dataset))

def get_denoiser(config, model, diffusion):
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

def get_batch_x_t(config, batch_imgs, sigmas, sampler):
    '''
    Return batch of x_t with t sampled randomly. This function
    follows the sampling scheme of sigma_i from Improved Techniques,
    but could be converted to sample sigma uniformly. 
    '''
    assert config.exp.batch_size >= config.exp.n_samples
    timesteps, batch_weights = sampler.sample(config.exp.batch_size)
    batch_sigmas = sigmas.gather(dim = 0, index = timesteps)
    batch_x_t = batch_imgs + batch_sigmas * torch.randn_like(batch_imgs)
    return batch_x_t, batch_sigmas, batch_weights

def run(config: DictConfig):
    utils.preprocess_config(config)
    utils.setup_wandb(config)

    # make fabric
    fabric = get_fabric(config)

    # setup components, optimizer and dataloader
    corrector, optimizer, cm, diffusion, sampler = get_components(config, fabric)
    dataloader = get_dataloader(config, fabric)

    # automatically move each created tensor to proper device
    with fabric.init_tensor():
        
        # get sigmas for sampling
        sigmas = utils.get_sigmas_karras(config)
        sigma_max = utils.get_sigma_max(config)

        for batch_idx, batch_x_0 in enumerate(dataloader):
            log.info(f'Batch index: {batch_idx}')
            optimizer.zero_grad()
            
            # sample noise with random sigma to get x_t
            batch_x_t, batch_sigmas, batch_weights = get_batch_x_t(config, batch_x_0, sigmas, sampler)

            # get approximate x_0 with cm
            denoiser = get_denoiser(config, cm, diffusion)
            batch_x_0_hat = denoiser(batch_x_t, batch_sigmas)

            # enhance x_0 with corrector
            batch_c_input = torch.cat([batch_x_0_hat, batch_x_t])
            batch_x_0_hat_c = corrector(batch_c_input, batch_sigmas)

            # regress wrt to x_0
            # TODO: batch_weights is incorrect since we sample timesteps
            #       uniformly but sigmas have non-uniform distribution
            loss = F.mse_loss(batch_x_0, batch_x_0_hat_c) # * batch_weights
            fabric.backward(loss)
            optimizer.step()
            
            log.info(f'Loss: {loss.mean().item()}')
            import pdb; pdb.set_trace()
