import torch
import wandb
import omegaconf
from torch import nn
from pathlib import Path

class HMove(nn.Module):

    def __init__(
            self, 
            shape: str|list, 
            norm_scale: float,
            id: str, 
            step_size: list, 
            use: bool,
            use_fp16: bool,
            print_shapes: bool,
            path_save: str,
            path_load: str|None):
        '''
        shape - shape of the direction along which to move.
            Provided as a list representing the direction's shape.
        norm_scale - coefficient that scales the norm of the direction which
            is initially equal to the norm of the representation.
        id - identifier of the layer/block which representation will be
            moved along the direction.
        step_size - step size which scales the normalized direction.
            If list, it defines minimum and maximum step size used
                when moving along a direction. Must have length 2.
        use - boolean indicating whether to move in any direction or not.
            If True, the h move is performed.
            If False, the object is initialized in a way that does not
                influence any outputs.
        print_shapes - boolean indicating whether to print shapes of
            each block's representation. Useful when trying to find the
            shape for config.
        path_save - path where the object's checkpoint will be saved.
        path_load - optional path from which the object will be loaded.
        '''
        super().__init__()
        assert isinstance(step_size, omegaconf.listconfig.ListConfig) and len(step_size) == 2
        self.norm_scale = torch.tensor(norm_scale)
        self.step_size = step_size
        self.use_fp16 = use_fp16
        self.use = use
        self.id = id
        self.register_buffer('dir', self.get_dir(shape))
        self.load(path_load)
        self.path_save = Path(path_save)
        self.print_shapes = print_shapes
        self.save()

    def save(self):
        # make save path
        path_save = self.path_save / 'wandb' / 'latest-run' / 'artifacts'
        path_save.mkdir(parents = True, exist_ok = True)
        path_save = path_save / 'h_move.pt'
        
        # save ckpt to it
        torch.save(self.state_dict(), path_save)

        # create wandb artifact and add ckpt
        art = wandb.Artifact(name = 'h_move', type = 'model')
        art.add_file(path_save)

    def load(self, path_load):
        if path_load is not None:
            self.load_state_dict(torch.load(path_load))

    def get_dir(self, shape):
        '''
        Returns the direction based on provided shape. 
        '''
        dir = torch.randn(1, *shape)
        dir /= dir.norm()

        if self.use_fp16:
            dir = dir.half()

        return dir
        
    def move(self, h, block_id):
        assert block_id is not None, 'Block must be assigned an identifier'

        if self.print_shapes:
            print(f'["{block_id}"]="{list(h.shape)[1:]}" \\')

        if self.use and block_id == self.id:
            # creates a sequence of uniformly spaced step 
            # of length equal to batch size sizes
            step_size = torch.linspace(*self.step_size, h.shape[0])

            if self.use_fp16:
                step_size = step_size.half()

            dims_to_add = self.dir.ndim - step_size.ndim
            step_size = step_size.view((-1,) + dims_to_add * (1,))
            h_norm = h[0].norm()

            return h + step_size * self.norm_scale * h_norm * self.dir
        
        else:
            return h

    def forward(self, x):
        pass

    