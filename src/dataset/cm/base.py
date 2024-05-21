import math
import random
import numpy as np
from PIL import Image
from abc import ABC
from torch.utils.data import Dataset

class RealDataDataset(ABC, Dataset):

    def center_crop_arr(self, pil_image, image_size):
        '''
        Ported from original CM repository.
        '''
        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - image_size) // 2
        crop_x = (arr.shape[1] - image_size) // 2
        return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

    def random_crop_arr(self, pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
        '''
        Ported from original CM repository.
        '''
        min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
        max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
        smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * smaller_dim_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = smaller_dim_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image)
        crop_y = random.randrange(arr.shape[0] - image_size + 1)
        crop_x = random.randrange(arr.shape[1] - image_size + 1)
        return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
