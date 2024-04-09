import torch

class StandardNormalSingleNoiseDataset(torch.utils.data.Dataset):

    def __init__(self, n_samples: int, resolution: int, seed: int):
        super().__init__()
        self.data = torch.randn(
            1, 3, resolution, resolution, generator = torch.Generator().manual_seed(seed))
        self.length = n_samples

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.data[0]
        