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

    # if "consistency" in args.training_mode:
    #     distillation = True
    # else:
    #     distillation = False

    # model, diffusion = create_model_and_diffusion(
    #     **args_to_dict(args, model_and_diffusion_defaults().keys()),
    #     distillation=distillation,
    # )
    model = fabric.setup(instantiate(config.model.model))
    import pdb; pdb.set_trace()
    diffusion = instantiate(config.model.diffusion)
    return model, diffusion

def get_dataloader(config, fabric):
    return fabric.setup_dataloaders(instantiate(config.dataset))

def run(config: DictConfig):
    utils.preprocess_config(config)
    utils.setup_wandb(config)

    fabric = get_fabric(config)

    model, diffusion = get_components(config, fabric)

    dataloader = get_dataloader(config, fabric)

   