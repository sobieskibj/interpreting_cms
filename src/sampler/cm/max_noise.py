import numpy as np
import torch

@torch.no_grad()
def max_noise(
    distiller, x, sigmas, k, t_min = 0.002, t_max = 80.0, rho = 7.0, steps = 40, fix_noise = False):

    s_in = x.new_ones([x.shape[0]])
    xs = []
    
    if fix_noise:
        # fixes the noise to always move along the same trajectory
        noise  = x.clone()

    # scale to proper std
    x =  x * sigmas[0]

    for _ in range(k):
        x0 = distiller(x, t_max * s_in)

        if not fix_noise:
            noise = torch.randn_like(x)

        xs.append((x0 + 1) / 2)
        x = x0 + noise * np.sqrt(t_max ** 2 - t_min ** 2)

    return torch.cat(xs, 0)