import torch

from model.cm.src.samplers import append_zero

def get_sampler_name(sampler_target):
    return sampler_target.split('.')[-1]

def get_sigmas_karras(config):
    '''Constructs the noise schedule of Karras et al. (2022).'''
    steps = config.sampler.steps
    sampler_name = get_sampler_name(config.sampler._target_)
    n = steps + 1 if sampler_name == 'progdist' else steps
    ramp = torch.linspace(0, 1, n)
    rho = config.diffusion.rho
    min_inv_rho = config.diffusion.sigma_min ** (1 / rho)
    max_inv_rho = config.diffusion.sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas)

def get_sigmas_karras_(steps, rho, sigma_min, sigma_max):
    ramp = torch.linspace(0, 1, steps)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas

def is_class_cond(config):
    '''
    Use class conditioning if num_classes was provided
    '''
    return config.model.num_classes is not None

def get_sigma_max(config):
    return config.diffusion.sigma_max