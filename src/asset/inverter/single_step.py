import torch
import torch.nn.functional as F
from tqdm import tqdm

from .base import BaseInverter


import logging
log = logging.getLogger(__name__)


class SingleStepInverter(BaseInverter):


    def __init__(
            self, 
            optimizer_partial: torch.optim.Optimizer,
            loss_stop_crit: float,
            n_iters: int,
            sigmas: torch.Tensor,
            eval_ts: list[int],
            t: int):
        
        super().__init__()
        self.optimizer_partial = optimizer_partial
        self.loss_stop_crit = loss_stop_crit
        self.n_iters = n_iters
        self.register_buffer('sigmas', sigmas.flip(0))
        self.register_buffer('sigma_min', sigmas.min())
        assert self.sigma_min > 0
        self.eval_ts = [e - 1 for e in eval_ts]
        self.t = t - 1

        self.stop = False
        self.ones = None


    def make_noise(self, x):
        # sample noise from standard gaussian and set it
        # as optimizer's parameters
        noise = torch.randn_like(x, requires_grad = True)
        self.optimizer = self.optimizer_partial(params = [noise])

        # in the beginning, we create ones for model's input
        if self.ones is None:
            self.ones = torch.ones(x.shape[0])

        return noise
    

    def get_noise_scale(self, sigma_t):
        return torch.sqrt(sigma_t ** 2 - self.sigma_min ** 2)


    def denoise(self, denoiser, noise, x_0):
        sigma_t = self.sigmas[self.t]
        c = self.get_noise_scale(sigma_t)
        x_t = x_0 + noise * c
        x_0_hat = denoiser(x_t, self.ones * sigma_t)
        return x_0_hat
    

    def step(self, batch_x, batch_x_hat):
        # gradient step
        loss = F.mse_loss(batch_x, batch_x_hat)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        loss = loss.item()

        # NOTE: this depends on batch size, i.e. stops if 
        #       the _combined_ loss satisfies the criterion
        if loss < self.loss_stop_crit:
            self.stop = True

        return {'losses/train': loss}


    def invert(self, batch_x_0, denoiser):
        # make initial noise
        batch_noise = self.make_noise(batch_x_0)
        
        # iteratively invert
        pbar = tqdm(range(self.n_iters))
        for iter in pbar:
            batch_x_0_hat = self.denoise(denoiser, batch_noise, batch_x_0)
            loss_dict = self.step(batch_x_0, batch_x_0_hat)
            loss = loss_dict["losses/train"]
            pbar.set_description(f"{loss=}")

        log.info(f"Terminating inversion at {iter=} with {loss=}")

        # return noise that inverts the image
        return batch_noise


    @torch.no_grad()
    def evaluate(self, denoiser, noise, x_0):
        batch_size = x_0.shape[0]
        n_eval_steps = len(self.eval_ts)
        
        # get sigmas and scales
        sigma_ts = self.sigmas[self.eval_ts].repeat(batch_size)
        cs = self.get_noise_scale(sigma_ts).view(-1, 1, 1, 1)
        
        # repeat
        x_0 = x_0.repeat_interleave(n_eval_steps, dim = 0)
        noise = noise.repeat_interleave(n_eval_steps, dim = 0)
        ones = torch.ones(noise.shape[0])

        # noise x_0 and denoise
        x_t = x_0 + noise * cs
        x_0_hat = denoiser(x_t, ones * sigma_ts)                

        # evaluate
        loss = F.mse_loss(x_0, x_0_hat).item()

        # return dicts of results
        scalars = {'losses/eval': loss}
        imgs = {'images/eval_reconstruction': x_0_hat}

        return scalars, imgs


    @property
    def stop(self):
        return self.stop_opt
    
    
    @stop.setter
    def stop(self, v):
        self.stop_opt = v
    