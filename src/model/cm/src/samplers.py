import numpy as np
import torch
import torchvision

def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]

def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)

def get_ancestral_step(sigma_from, sigma_to):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    sigma_up = (
        sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2
    ) ** 0.5
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


@torch.no_grad()
def sample_euler_ancestral(model, x, sigmas):
    """Ancestral sampling with Euler method steps."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)

    for i in indices:
        denoised = model(x, sigmas[i] * s_in)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1])

        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        x = x + torch.randn_like(x) * sigma_up
    return x


@torch.no_grad()
def sample_midpoint_ancestral(model, x, ts):
    """Ancestral sampling with midpoint method steps."""
    s_in = x.new_ones([x.shape[0]])
    step_size = 1 / len(ts)

    for tn in ts:
        dn = model(x, tn * s_in)
        dn_2 = model(x + (step_size / 2) * dn, (tn + step_size / 2) * s_in)
        x = x + step_size * dn_2

    return x


@torch.no_grad()
def sample_heun(denoiser, x, sigmas, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, fix_noise = False):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)

    for i in indices:
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        if fix_noise:
            # repeats the same noise for the entire batch
            noise  = torch.randn_like(x[0]).unsqueeze(0).repeat_interleave(x.shape[0], dim = 0)
        else:
            noise = torch.randn_like(x)
        eps = noise * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)

        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    return x


@torch.no_grad()
def sample_euler(denoiser, x, sigmas):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)

    for i in indices:
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in)
        d = to_d(x, sigma, denoised)

        dt = sigmas[i + 1] - sigma
        x = x + d * dt
    return x


@torch.no_grad()
def sample_dpm(denoiser, x, sigmas, s_churn = 0.0, s_tmin = 0.0, s_tmax = float("inf"), s_noise = 1.0):
    """A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)

    for i in indices:
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)

        # Midpoint method, where the midpoint is chosen according to a rho=3 Karras schedule
        sigma_mid = ((sigma_hat ** (1 / 3) + sigmas[i + 1] ** (1 / 3)) / 2) ** 3
        dt_1 = sigma_mid - sigma_hat
        dt_2 = sigmas[i + 1] - sigma_hat
        x_2 = x + d * dt_1
        denoised_2 = denoiser(x_2, sigma_mid * s_in)
        d_2 = to_d(x_2, sigma_mid, denoised_2)
        x = x + d_2 * dt_2
    return x

### CM samplers ###

@torch.no_grad()
def sample_onestep(distiller, x, sigmas, steps = 40):
    """Single-step generation from a distilled model."""
    s_in = x.new_ones([x.shape[0]])
    import pdb; pdb.set_trace()
    return distiller(x, sigmas[0] * s_in)


@torch.no_grad()
def stochastic_iterative_sampler(
    distiller, x, sigmas, ts, t_min = 0.002, t_max = 80.0, rho = 7.0, steps = 40, fix_noise = False):

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)

        if fix_noise:
            # repeats the same noise for the entire batch
            noise  = torch.randn_like(x[0]).unsqueeze(0).repeat_interleave(x.shape[0], dim = 0)
        else:
            noise = torch.randn_like(x)

        x = x0 + noise * np.sqrt(next_t**2 - t_min**2)

    return x


@torch.no_grad()
def max_noise_sampler(
    distiller, x, sigmas, k, t_min = 0.002, t_max = 80.0, rho = 7.0, steps = 40, fix_noise = False):

    s_in = x.new_ones([x.shape[0]])
    xs = []
    
    if fix_noise:
        # NOTE: samplers could be refactored so that they scale 
        #   the input x inside instead of outside
        # fixes the noise to always move along the same trajectory
        noise  = x / sigmas[0]

    for _ in range(k):
        x0 = distiller(x, t_max * s_in)

        if not fix_noise:
            noise = torch.randn_like(x)

        xs.append((x0 + 1) / 2)
        x = x0 + noise * np.sqrt(t_max ** 2 - t_min ** 2)

    return torch.cat(xs, 0)

### Progressive Distillation ###

@torch.no_grad()
def sample_progdist(denoiser, x, sigmas):
    s_in = x.new_ones([x.shape[0]])
    sigmas = sigmas[:-1]  # skip the zero sigma

    indices = range(len(sigmas) - 1)

    for i in indices:
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in)
        d = to_d(x, sigma, denoised)
        dt = sigmas[i + 1] - sigma
        x = x + d * dt

    return x
