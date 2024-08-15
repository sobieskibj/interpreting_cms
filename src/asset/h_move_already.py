import torch
import wandb
import omegaconf
from torch import nn
from pathlib import Path

class HMoveAlready(nn.Module):

    def __init__(
            self, 
            shape: str|list, 
            id: str, 
            use: bool,
            use_fp16: bool,
            print_shapes: bool,
            path_save: str,
            path_load: str|None):
        '''
        shape - shape of the direction along which to move.
            Provided as a list representing the direction's shape.
        id - identifier of the layer/block which representation will be
            moved along the direction.
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
        self.use_fp16 = use_fp16
        self.use = use
        self.id = id
        self.dir = torch.nn.Parameter(self.get_dir(shape))
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
        dir = torch.zeros(1, *shape)

        if self.use_fp16:
            dir = dir.half()

        return dir
        

    def move(self, h, block_id):
        assert block_id is not None, 'Block must be assigned an identifier'

        if self.print_shapes:
            print(f'["{block_id}"]="{list(h.shape)[1:]}" \\')

        if self.use and block_id == self.id:
            return h + self.dir
        
        else:
            return h


    def forward(self, x):
        pass

    