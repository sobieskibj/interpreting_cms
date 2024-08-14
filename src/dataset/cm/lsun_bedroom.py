
import torch
from datasets import load_dataset
from torchvision.transforms.functional import pil_to_tensor


class LSUNBedroom(torch.utils.data.Dataset):


    def __init__(self, split: str):
        super().__init__()
        self.data = load_dataset("pcuenq/lsun-bedrooms", split=split)
        self.length = len(self.data)


    def __len__(self):
        return self.length
    

    def __getitem__(self, idx):
        img_pil = self.data[idx]["image"]
        img_tensor = pil_to_tensor(img_pil)
        return img_tensor, img_pil