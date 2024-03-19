import wandb
import omegaconf
from omegaconf import DictConfig

def setup_wandb(config: DictConfig):
    # extract two last subdirs
    group_name = config.exp.log_dir.relative_to(config.exp.log_dir.parents[1])
    # split into group and name
    group, name = group_name.parts
    wandb_config = omegaconf.OmegaConf.to_container(
        config, resolve = True, throw_on_missing = True)
    wandb.init(
        project = config.wandb.project,
        dir = config.exp.log_dir,
        group = group,
        name = name,
        config = wandb_config,
        sync_tensorboard = True)