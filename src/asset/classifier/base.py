from abc import ABC, abstractmethod
from torch import nn

class ClassifierBase(ABC, nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval()