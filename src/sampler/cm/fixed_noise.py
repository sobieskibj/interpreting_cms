import numpy as np
import torch

@torch.no_grad()
def fixed_noise(
        distiller, x, sigmas, k, t_step, t_min = 0.002, t_max = 80.0, rho = 7.0, steps = 40, 
        fix_noise = False, correction = True, noiser = False):
    '''
    t_step - timestep indicating the fixed noise scale
    k - number of iterations on a fixed noise scale
    noiser - start with image
    '''

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    xs = [None] * (k + 1)

    x0 = x
    if not noiser:
        if fix_noise:
            # fixes the noise to always move along the same trajectory
            noise  = x.clone()

        # scale to proper std
        x =  x * sigmas[0]

        # obtain initial x0 from full noise
        x0 = distiller(x, t_max * s_in)
    xs[0] = x0

    # iterate using a fixed noise scale
    t = (t_min_rho + t_step / (steps - 1) * (t_max_rho - t_min_rho)) ** rho

    for k_step in range(1, k + 1):

        if not fix_noise:
            # if trajectory is not fixed, sample a random one
            noise = torch.randn_like(x)

        # either use a corrected scale or not
        if correction:
            x = x0 + noise * np.sqrt(t ** 2 - t_min ** 2)
        else:
            x = x0 + noise * t

        # denoise and save
        x0 = distiller(x, t * s_in)
        xs[k_step] = x0

    return torch.cat(xs, 0)
