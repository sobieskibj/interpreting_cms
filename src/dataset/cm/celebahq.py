from .base import RealDataDataset

import random
import numpy as np
import torchvision
from pathlib import Path

class CelebAHQDataset(RealDataDataset):

    def __init__(self, path_data: str, img_size: int, n_samples: int, random_crop: bool, random_flip: bool):
        super().__init__()
        self.path_data = Path(path_data)
        self.img_size = img_size
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.length = n_samples

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        path_img = self.path_data / str(idx) / "img.png"
        img = torchvision.io.read_image(str(path_img))
        img = torchvision.transforms.functional.to_pil_image(img)
        pil_image = img.convert("RGB")

        if self.random_crop:
            arr = self.random_crop_arr(pil_image, self.img_size)
        else:
            arr = self.center_crop_arr(pil_image, self.img_size)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1
        
        # NOTE: we assume no class conditioning for now
        # out_dict = {}
        # if self.local_classes is not None:
        #     out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        return np.transpose(arr, [2, 0, 1]), {}
    