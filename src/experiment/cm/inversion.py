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


def update_pbar(pbar, loss, eval_loss):
    pbar.set_description(f'loss: {loss}, evaluation loss: {eval_loss}')
    wandb.log({
        'losses/reconstruction': loss,
        'losses/eval_reconstruction': eval_loss})


def log_every_n(config, step, x, name):
    if step % config.exp.log_every_n == 0:
        utils.log_img(x, name)


def run(config: DictConfig):
    utils.preprocess_config(config)
    utils.setup_wandb(config)

    # make fabric
    fabric = get_fabric(config)

    # setup components and dataloader
    model, diffusion, inverter = get_components(config, fabric)
    dataloader = get_dataloader(config, fabric)

    # automatically move each created tensor to proper device
    with fabric.init_tensor():

        # create denoiser with no class conditioning
        denoiser = get_denoiser(config, model, diffusion)

        for batch_idx, batch_x in enumerate(dataloader):

            log.info(f'Batch index: {batch_idx}')
            utils.log_img(batch_x, 'images/gt')

            # make initial noise to optimize
            batch_noise = inverter.make_noise(batch_x)
            inv_pbar = tqdm(range(config.exp.n_inv_steps))

            for inv_step in inv_pbar:

                # get model predictions
                batch_x_hat = inverter.denoise(denoiser, batch_noise, batch_x)
                log_every_n(config, inv_step, batch_x_hat, 'images/reconstruction')

                # make gradient step in noise 
                loss = inverter.step(batch_x, batch_x_hat)

                # run evaluation
                if inv_step % config.exp.eval_every_n == 0:
                    eval_loss, batch_x_hat_eval = inverter.evaluate(denoiser, batch_noise, batch_x)
                    utils.log_img(batch_x_hat_eval, 'images/eval_reconstruction')

                # log losses
                update_pbar(inv_pbar, loss, eval_loss)

                # check if stopping criterion is achieved
                if inverter.stop:
                    log.info('Stopping criterion fulfilled')
                    break

            inv_pbar.close()
