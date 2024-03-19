import torch

class StandardNormalNoiseDataset(torch.utils.data.Dataset):

    def __init__(self, n_samples: int, resolution: int):
        super().__init__()
        
        self.data = torch.randn(n_samples, 3, resolution, resolution)
        self.length = n_samples

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.data[index]
        