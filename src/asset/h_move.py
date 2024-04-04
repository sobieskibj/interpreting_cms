import torch
import omegaconf
from torch import nn

class HMove(nn.Module):

    def __init__(
            self, 
            source: str|list, 
            id: str, 
            step_size: list, 
            use: bool,
            use_fp16: bool):
        '''
        source - source of the direction along which to move.
            If str, it is treated as path to .pt file.
            If list, it represents the shape of a randomly sampled direction.
        id - identifier of the layer/block which representation will be
            moved along the direction.
        step_size - step size which scales the normalized direction.
            If list, it defines minimum and maximum step size used
                when moving along a direction. Must have length 2.
        use - boolean indicating whether to move in any direction or not.
            If True, the h move is performed.
            If False, the object is initialized in a way that does not
                influence any outputs.
        '''
        super().__init__()
        assert isinstance(step_size, omegaconf.listconfig.ListConfig) and len(step_size) == 2
        self.id = id
        self.step_size = step_size
        self.use_fp16 = use_fp16
        self.register_buffer('dir', self.get_dir(source, use))

    def get_dir(self, source, use):
        '''
        Returns the direction based on provided source. 
        '''
        if use:
            if isinstance(source, str):
                state_dict = torch.load(source)
                dir = state_dict['dir']
                assert self.id == state_dict['id']
                import pdb; pdb.set_trace()
                # make sure that it is loaded correctly

            elif isinstance(source, omegaconf.listconfig.ListConfig):
                dir = torch.randn(1, *source)
                dir /= dir.norm()

            if self.use_fp16:
                dir = dir.half()

            return dir

        else:
            return None
        
    def move(self, h, block_id):
        assert block_id is not None, 'Block must be assigned an identifier'

        if self.dir is not None and block_id == self.id:
            # creates a sequence of uniformly spaced step 
            # of length equal to batch size sizes
            step_size = torch.linspace(*self.step_size, h.shape[0])

            if self.use_fp16:
                step_size = step_size.half()

            dims_to_add = self.dir.ndim - step_size.ndim
            step_size = step_size.view((-1,) + dims_to_add *(1,))

            return h + step_size * self.dir
        
        else:
            return h

    def forward(self, x):
        pass

    