import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

from .base import ClassifierBase

class ResNet50(ClassifierBase):
    def __init__(self):
        super().__init__()

        wgts = ResNet50_Weights.IMAGENET1K_V1
        self.id_to_cls = wgts.meta['categories']
        self.model = resnet50(weights = wgts)
        self.transforms = wgts.transforms()

    def forward(self, x):
        # NOTE: Input is required to be in [0, 1] range
        x = self.transforms(x)
        x = self.model(x)
        return x