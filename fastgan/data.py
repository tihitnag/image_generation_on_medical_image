import numpy as np
import torch.utils.data as data

from PIL import Image

from fastgan.utils import parse_image_set

"""
Infinite Sampler and InfiniteSamplerWrapper taken from:
https://github.com/odegeasslbc/FastGAN-pytorch/blob/main/operation.py
"""

def get_dataloader(dataset, batch_size, dataloader_workers=4):
    return iter(data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                sampler=InfiniteSamplerWrapper(dataset), #num_workers=dataloader_workers,
                                pin_memory=True))


def InfiniteSampler(n):
    """Data sampler"""
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    """Data sampler wrapper"""

    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


class ImageFolder(data.Dataset):
    def __init__(self, image_set, transforms=None):
        super().__init__()
        self.image_files = parse_image_set(image_set)
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        try:
            image = Image.open(img_name).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_name}: {str(e)}")
            # Return a black image of size 256x256 as fallback
            image = Image.new('RGB', (256, 256), (0, 0, 0))

        if self.transforms:
            image = self.transforms(image)

        return image


class IndividualKLGradeImageFolder(ImageFolder):
    def __init__(self, image_set, kl_grade, transforms=None):
        super().__init__(image_set, transforms)
        print(f"Restricted Domain to {kl_grade} images.")
        print(f"Original size: {len(self.image_files)}")
        if kl_grade == "normal":
            self.image_files = [img for img in self.image_files if "normal" in img]
        elif kl_grade == "abnormal":
            self.image_files = [img for img in self.image_files if "abnormal" in img]
        else:
            raise ValueError(f"Invalid grade: {kl_grade}. Must be 'normal' or 'abnormal'")
        print(f"New size: {len(self.image_files)}")

