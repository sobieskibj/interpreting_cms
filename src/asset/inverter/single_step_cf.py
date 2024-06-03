import torch
import torch.nn.functional as F

from .base import BaseInverter
from asset.classifier import ClassifierBase

import logging
log = logging.getLogger(__name__)


class SingleStepCounterfactualInverter(BaseInverter):


    def __init__(
            self, 
            classifier: ClassifierBase,
            target_class: int,
            alpha: float,
            optimizer_partial: torch.optim.Optimizer,
            stop_crit: float,
            sigmas: torch.Tensor,
            eval_ts: list[int],
            t: int):
        
        super().__init__()

        self.clf = classifier
        self.target_class = target_class
        
        self.optimizer_partial = optimizer_partial
        self.stop_crit = stop_crit
        self.alpha = alpha

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

        # we also save classifier's predictions to check at eval
        with torch.no_grad():
            self.preds = F.softmax(self.clf(normalize(x)), dim = -1).argmax(dim = 1)
            is_target = (self.preds == self.target_class).sum().item()
            
            if is_target > 0:
                log.info(f'Found {is_target} predictions with target class')

        return noise
    

    def get_noise_scale(self, sigma_t):
        return torch.sqrt(sigma_t ** 2 - self.sigma_min ** 2)


    def get_loss(self, x, x_hat):
        x, x_hat = normalize(x), normalize(x_hat)
        # NOTE: softmax probably lowers the signal strength
        prob = F.softmax(self.clf(x_hat), dim = -1).\
            select(dim = 1, index = self.target_class).view(-1, 1)
        return F.mse_loss(x, x_hat) - self.alpha * prob.mean()


    def denoise(self, denoiser, noise, x_0):
        self.optimizer.zero_grad()
        sigma_t = self.sigmas[self.t]
        c = self.get_noise_scale(sigma_t)
        x_t = x_0 + noise * c
        x_0_hat = denoiser(x_t, self.ones * sigma_t)
        return x_0_hat
    

    def step(self, batch_x, batch_x_hat):
        # gradient step
        loss = self.get_loss(batch_x, batch_x_hat)
        loss.backward()
        self.optimizer.step()

        loss = loss.item()

        # NOTE: this depends on batch size, i.e. stops if and
        #       and only if the _combined_ loss satisfies the criterion
        if loss < self.stop_crit:
            self.stop = True

        return loss


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

        # check how many predictions changed to target class
        preds = F.softmax(self.clf(normalize(x_0_hat)), dim = -1).argmax(dim = 1)
        flipped = preds == self.target_class
        fr = (flipped * 1.).mean().item()
        log.info(f'flip rate: {fr}')

        # evaluate
        loss = self.get_loss(x_0, x_0_hat)

        # zero out images which were not flipped
        x_0_hat[~flipped] = 0.

        return loss.item(), x_0_hat


    @property
    def stop(self):
        return self.stop_opt
    
    
    @stop.setter
    def stop(self, v):
        self.stop_opt = v


def normalize(x):
    x = x - x.flatten(start_dim = 1).min(dim = 1)[0].view(-1, 1, 1, 1)
    x = x / x.flatten(start_dim = 1).max(dim = 1)[0].view(-1, 1, 1, 1)
    return x