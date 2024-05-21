import wandb
import omegaconf
from omegaconf import DictConfig
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

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
    
def watch_models(*models):
    wandb.watch(models = models)

def log_grid(imgs, name):
    grid = make_grid(imgs).permute(1, 2, 0)
    grid = wandb.Image(grid.numpy(force = True))
    wandb.log({name: grid})

def log_img(x, name):
     if x.shape[0] == 1:
         wandb.log({name: wandb.Image(x)})
     else:
         # we manually normalize each image in a batch
         # to [0, 1] range. for single image, this is
         # not needed
         x = make_grid(x, scale_each = True, normalize = True)
         x = to_pil_image(x)
         wandb.log({name: wandb.Image(x)})