from torch import nn
from abc import ABC, abstractmethod

class BaseInverter(ABC, nn.Module):

    @abstractmethod
    def make_noise(self, x):
        raise NotImplementedError
        
    @abstractmethod
    def denoise(self, sampler, denoiser, batch_noise, batch_x):
        raise NotImplementedError
    
    @abstractmethod
    def step(self, batch_x, batch_x_hat, batch_noise):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, sampler, denoiser, batch_noise, batch_x):
        raise NotImplementedError

    @property
    @abstractmethod
    def stop(self):
        raise NotImplementedError