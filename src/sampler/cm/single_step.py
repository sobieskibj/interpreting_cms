import torch

@torch.no_grad()
def single_step(distiller, x, sigmas, steps = 40):
    """
    Single-step generation from a distilled model.
    
    Modified to change std of x at the sampler level and not on the outside.
    """
    s_in = x.new_ones([x.shape[0]])
    return distiller(x * sigmas[0], sigmas[0] * s_in)