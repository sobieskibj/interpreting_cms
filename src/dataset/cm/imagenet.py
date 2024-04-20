from .base import RealDataDataset

import random
import numpy as np
from datasets import load_dataset

class ImageNetDataset(RealDataDataset):

    def __init__(self, split: str):
        super().__init__()
        self.split = split
        self.img_size = 256
        self.data = load_dataset('imagenet-1k', split = self.split, trust_remote_code = True)
        self.length = min(len(self.data))

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img, label = item['image'], item['label']
        pil_image = img.convert("RGB")

        if self.random_crop:
            arr = self.random_crop_arr(pil_image, self.resolution)
        else:
            arr = self.center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        # NOTE: we assume no class conditioning for now
        
        # out_dict = {}
        # if self.local_classes is not None:
        #     out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        return np.transpose(arr, [2, 0, 1])
    