import wandb
import torch
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig
from hydra.utils import instantiate

import utils

import logging
log = logging.getLogger(__name__)

torch.autograd.set_detect_anomaly(True)


def get_fabric(config):
    fabric = instantiate(config.fabric)
    fabric.seed_everything(config.exp.seed)
    fabric.launch()
    return fabric


def get_components(config, fabric):
    model = fabric.setup(instantiate(config.model))
    diffusion = instantiate(config.diffusion)
    inverter = fabric.setup(instantiate(config.inverter))
    already_loss = fabric.setup(instantiate(config.already_loss))
    already_loss.eval().requires_grad_(False)
    h_move = instantiate(config.h_move)
    optimizer_partial = instantiate(config.optimizer)
    optimizer = optimizer_partial(params=h_move.parameters())
    h_move, optimizer = fabric.setup(h_move, optimizer)
    return model, diffusion, inverter, already_loss, h_move, optimizer


def get_dataloader(config, fabric):
    return fabric.setup_dataloaders(instantiate(config.dataset))


def get_denoiser(config, model, diffusion, h_move):

    model_kwargs = {"h_move": h_move}

    def denoiser(x_t, sigma):
        _, denoised = diffusion.denoise(model, x_t, sigma, **model_kwargs)
        return denoised
    
    return denoiser


def from_01_to_m1p1(x):
    return (x - 0.5) * 2


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def run(config: DictConfig):
    utils.preprocess_config(config)
    utils.setup_wandb(config)

    # make fabric
    fabric = get_fabric(config)

    # setup components and dataloader
    model, diffusion, inverter, already_loss, h_move, optimizer = get_components(config, fabric)
    dataloader = get_dataloader(config, fabric)

    # create dir for dumping noise for inversion
    path_dump_noise = Path(config.exp.path_dump_noise)
    path_dump_noise.mkdir(parents=True, exist_ok=True)

    # automatically move each created tensor to proper device
    with fabric.init_tensor():

        for batch_id, batch_x in enumerate(dataloader):

            # unpack batch and check if noises are saved
            batch_idx, batch_x_tensor, batch_x_pil = batch_x
            wandb.log({"images/original": wandb.Image(batch_x_tensor)})
            is_dumped = all([(path_dump_noise / f"{img_id}.pt").exists() for img_id in batch_idx])

            # create denoiser with disabled h_move
            h_move.use = False
            denoiser = get_denoiser(config, model, diffusion, h_move)

            if is_dumped:
                log.info("Skipping inversion")
                # load batch of noises if they are all dumped
                batch_noise = torch.stack([torch.load(path_dump_noise / f"{img_id}.pt") for img_id in batch_idx])

            else:
                # find noise that inverts the original image
                log.info("Inverting the images")
                batch_noise = inverter.invert(from_01_to_m1p1(batch_x_tensor), denoiser)
                for img_id, noise in zip(batch_idx, batch_noise):
                    torch.save(noise, path_dump_noise / f"{img_id}.pt")

            # invert the image for logging
            with torch.no_grad():
                inverter.make_noise(batch_x_tensor)
                batch_x_inverted = inverter.denoise(denoiser, batch_noise, from_01_to_m1p1(batch_x_tensor))
                wandb.log({"images/inverted": wandb.Image(normalize(batch_x_inverted))})

            # create denoiser with enabled h_move
            h_move.use = True
            denoiser = get_denoiser(config, model, diffusion, h_move)

            # optimize h move wrt to already_loss
            log.info("Optimizing already_loss")
            log_every = config.exp.log_every
            text_target = config.exp.text_target
            text_source = config.exp.text_source
            
            pbar = tqdm(range(config.exp.n_iters))

            for iter in pbar:
                # gradient step
                optimizer.zero_grad()
                batch_x_edit = inverter.denoise(denoiser, batch_noise, from_01_to_m1p1(batch_x_tensor))
                loss = already_loss(
                    img_edit=normalize(batch_x_edit), 
                    img_source=batch_x_tensor, 
                    text_target=text_target, 
                    text_source=text_source)
                fabric.backward(loss)
                optimizer.step()

                pbar.set_description(f"already_loss={loss.item()}")

                if iter % log_every == 0:
                    wandb.log({"images/modified": wandb.Image(normalize(batch_x_edit))})