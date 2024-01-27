from typing import AnyStr
from argparse import Namespace
from glob import glob

from PIL import Image

import torchvision

torchvision.disable_beta_transforms_warning()

from torch.utils.data import Dataset
from torchvision.datasets.folder import is_image_file
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.v2 import (Compose, ColorJitter, Resize, CenterCrop,
                                       RandomResizedCrop, RandomAffine,
                                       ToImageTensor, ConvertImageDtype,
                                       ScaleJitter, RandomCrop, Normalize,
                                       RandomGrayscale, RandomAdjustSharpness,
                                       RandomRotation, RandomHorizontalFlip)

import utils
from utils.logging_tool import get_logger

from .utils import exists


class ImageDataset(Dataset):

    def __init__(self, mode: AnyStr, args: Namespace, data_root: AnyStr):
        super(ImageDataset, self).__init__()
        logger = get_logger()

        self.mode = mode

        samples = list(filter(lambda fname: is_image_file(fname), glob(f"{data_root}/**/*", recursive=True)))

        self.num_eval = min(int(args.data_spec.num_eval), int(len(samples) * 0.025))

        if mode == utils.mode.TRAIN:
            self.samples = samples[:-self.num_eval]
        elif mode == utils.mode.EVAL:
            self.samples = samples[-self.num_eval:]
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
                if exists('random_affine', aug_spec):
                    transforms.append(RandomAffine(**aug_spec['random_affine']))
                if exists('random_crop', aug_spec):
                    transforms.append(RandomCrop(**aug_spec['random_crop']))
                else:
                    transforms.append(Resize(**aug_spec.resize, antialias=True))
                    transforms.append(RandomCrop(**aug_spec.resize, pad_if_needed=True, padding_mode='reflect'))
                    # transforms.append(CenterCrop(**aug_spec.resize))
                    # transforms.append(RandomResizedCrop(**aug_spec.resize, antialias=True))
                    # transforms.append(RandomCrop(**aug_spec.resize, pad_if_needed=True, padding_mode='reflect'))
        else:
            transforms.append(Resize(**args.data_spec.aug.resize, antialias=True))
            transforms.append(CenterCrop(**args.data_spec.aug.resize))

        transforms.extend([ToImageTensor(), ConvertImageDtype()])

        if exists('normalize', args.data_spec.aug):
            transforms.append(Normalize(**args.data_spec.aug.normalize))

        self.transform = Compose(transforms)

        logger.info(f"Create {self.__str__()} {mode} dataset with {self.__len__()} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img = self.transform(Image.open(self.samples[index]).convert("RGB"))
        return img, img  # return identical image-pair

    def __str__(self):
        return self.__class__.__name__.lower()
