import torch
import numpy as np

@torch.no_grad()
def multi_step(
    distiller, x, sigmas, ts, t_min = 0.002, t_max = 80.0, rho = 7.0, steps = 40, fix_noise = False, correction = True):

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    xs = [None] * (len(ts) - 1)

    if fix_noise:
        noise = x.clone()

    # scale to proper std
    x =  x * sigmas[0]

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)

        if not fix_noise:
            noise = torch.randn_like(x)

        xs[i] = x0

        if correction:
            x = x0 + noise * np.sqrt(next_t**2 - t_min**2)
        else:
            x = x0 + noise * next_t

    return torch.cat(xs, 0)