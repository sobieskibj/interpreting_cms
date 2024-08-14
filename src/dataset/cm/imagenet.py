from .base import RealDataDataset

import random
import numpy as np
from datasets import load_dataset

class ImageNetDataset(RealDataDataset):

    def __init__(
            self, 
            split: str, 
            img_size: int, 
            n_samples: int, 
            random_crop: bool, 
            random_flip: bool,
            filter_class_idx: list):
        super().__init__()
        self.split = split
        self.img_size = img_size
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.data = load_dataset('imagenet-1k', split = self.split, trust_remote_code = True)
        self.length = min(len(self.data), n_samples)
        self.filter_data(filter_class_idx)

    def filter_data(self, filter_class_idx):

        if filter_class_idx is not None:
            filtered_idx = [i for i, e in enumerate(self.data['label']) if e in filter_class_idx]
            self.map_index = lambda x: filtered_idx[x]
            self.length = min(self.length, len(filtered_idx))

        else:
            self.map_index = lambda x: x

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        idx = self.map_index(idx)
        item = self.data[idx]
        img, label = item['image'], item['label']
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

        return np.transpose(arr, [2, 0, 1])
    