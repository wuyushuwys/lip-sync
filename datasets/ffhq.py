from typing import AnyStr
from argparse import Namespace
from glob import glob

from PIL import Image

import torchvision

torchvision.disable_beta_transforms_warning()

from torch.utils.data import Dataset
from torchvision.datasets.folder import is_image_file
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.v2 import (Compose, ColorJitter, Resize,
                                       ToImageTensor, ConvertImageDtype,
                                       ScaleJitter, RandomCrop, Normalize,
                                       RandomGrayscale, RandomAdjustSharpness,
                                       RandomRotation, RandomHorizontalFlip)

import utils.mode

from .utils import exists

DATA_ROOT = "data/FFHQ"


def get_dataset(mode: AnyStr, args: Namespace, data_root: AnyStr = DATA_ROOT):
    return FFHQ(mode, args, data_root)


class FFHQ(Dataset):

    def __init__(self, mode: AnyStr, args: Namespace, data_root: AnyStr):
        super(FFHQ, self).__init__()
        self.mode = mode
        self.num_eval = int(args.data_spec.num_eval)

        samples = list(filter(lambda fname: is_image_file(fname), glob(f"{data_root}/**/*", recursive=True)))

        self.samples = samples[:-self.num_eval]

        transforms = []
        if self.mode == utils.mode.TRAIN:
            if 'aug' in args.data_spec.keys():
                aug_spec = args.data_spec.aug
                if exists('rotate', aug_spec):
                    transforms.append(RandomRotation(**aug_spec['rotate'],
                                                     interpolation=InterpolationMode.BILINEAR,
                                                     fill=1))
                if exists('flip', aug_spec):
                    transforms.append(RandomHorizontalFlip(**aug_spec['flip']))
                if exists('grayscale', aug_spec):
                    transforms.append(RandomGrayscale(**aug_spec['grayscale']))
                if exists('sharpness', aug_spec):
                    transforms.append(RandomAdjustSharpness(**aug_spec['sharpness']))
                if exists('color_jitter', aug_spec):
                    transforms.append(ColorJitter(**aug_spec['color_jitter']))
                if exists('scale_jitter', aug_spec):
                    transforms.append(ScaleJitter(**aug_spec['scale_jitter'], antialias=True))
                if exists('random_crop', aug_spec):
                    transforms.append(RandomCrop(**aug_spec['random_crop']))
                else:
                    transforms.append(Resize(**aug_spec.resize, antialias=True))
        else:
            transforms.append(Resize(**args.data_spec.aug.resize, antialias=True))

        transforms.extend([ToImageTensor(), ConvertImageDtype()])
        if exists('normalize', args.data_spec.aug):
            transforms.append(Normalize(**args.data_spec.aug.normalize))

        self.transform = Compose(transforms)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img = self.transform(Image.open(self.samples[index]))
        # img = super().__getitem__(index)[0]
        return img, img  # return identical image-pair

    def __str__(self):
        return self.__class__.__name__.lower()
