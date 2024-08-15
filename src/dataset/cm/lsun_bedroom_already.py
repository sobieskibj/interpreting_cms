
import torch
from datasets import load_dataset
from torchvision.transforms.functional import pil_to_tensor, center_crop


def custom_collate_fn(batch):
    idx = [item[0] for item in batch]
    tensor_images = torch.stack([item[1] for item in batch])
    pil_images = [item[2] for item in batch]
    return idx, tensor_images, pil_images


class LSUNBedroomAlready(torch.utils.data.Dataset):


    def __init__(self, split: str, n_samples: int, img_size: int, n_skip: int):
        super().__init__()
        self.data = load_dataset("pcuenq/lsun-bedrooms", split=split)
        self.length = min(n_samples, len(self.data))
        self.img_size = img_size
        self.map_idx = lambda idx: idx + n_skip

    def __len__(self):
        return self.length
    

    def __getitem__(self, idx):
        idx = self.map_idx(idx)
        img_pil = self.data[idx]["image"].convert("RGB")
        img_tensor = pil_to_tensor(img_pil) / 255.
        assert self.img_size in img_tensor.shape
        img_tensor = center_crop(img_tensor, self.img_size)
        return [idx, img_tensor, img_pil]