import os
from typing import AnyStr, List
from argparse import Namespace
from glob import glob

import torch
from PIL import Image
import random
import torchvision

torchvision.disable_beta_transforms_warning()

from torch.utils.data import Dataset
from torchvision.datasets.folder import is_image_file
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.v2 import (Compose, Resize, CenterCrop,
                                       ToImageTensor, ConvertImageDtype, RandomCrop)
from torchvision.transforms.v2 import functional as tvf

import utils
from utils.logging_tool import get_logger


def make_dataset(dir, max_dataset_size=float("inf"), followlinks=True, method='glob'):
    if method == 'os':
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir, followlinks=followlinks)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        return images[:min(max_dataset_size, len(images))]
    elif method == 'glob':
        images = list(filter(lambda x: is_image_file(x), glob(os.path.join(dir, '**', '*'), recursive=True)))
        return images[:min(max_dataset_size, len(images))]
    else:
        raise NotImplementedError(f"{method} not implemented")

def random_resize(x, target_size=256):
    if torch.is_tensor(x):
        h, w = x.shape[1:]
    else:
        h, w = x.size
    if target_size <= h:
        scale_factor = random.randint(target_size, h) / h
        return tvf.resize(x, size=[int(h * scale_factor), int(w * scale_factor)],
                          interpolation=InterpolationMode.BICUBIC, antialias=True)
    else:
        return x


def random_flip(x, p=0.5):
    return tvf.horizontal_flip(x) if random.random() < p else x


def random_rotate(x, p=0.5):
    if random.random() < p:
        degree = random.choice([90, 270])
        return tvf.rotate(x, degree)
    return x


def normalize(x, mean, std):
    return tvf.normalize(x, mean=mean, std=std)


class ImageDataset(Dataset):

    def __init__(self, mode: AnyStr, args: Namespace, dataset: List[AnyStr]):
        super(ImageDataset, self).__init__()

        self.mode = mode

        self.samples = dataset

        self.use_rot = False
        self.use_flip = False
        self.use_random_size = False
        cropper = []
        to_tensor = []
        if self.mode == utils.mode.TRAIN:
            to_tensor.append(Resize(args.data_spec.train_size * 2, antialias=True))
            if args.data_spec.get('aug', False):
                if args.data_spec.aug.get("use_orig", False):
                    to_tensor[0] = RandomCrop(args.data_spec.train_size * 2, pad_if_needed=True, fill=1)
                self.use_rot = args.data_spec.aug.get("use_rot", False)
                self.use_flip = args.data_spec.aug.get("use_flip", False)
                self.use_random_size = args.data_spec.aug.get("use_random_size", False)
                self.gt_size = args.data_spec.train_size
            cropper.append(RandomCrop(size=self.gt_size, pad_if_needed=True, fill=1))

        else:
            cropper.append(Resize(**args.data_spec.aug.eval_size, antialias=True))
            cropper.append(CenterCrop(**args.data_spec.eval_size))

        to_tensor.extend([ToImageTensor(), ConvertImageDtype()])
        self.to_tensor = Compose(to_tensor)

        self.normalize = args.data_spec.get("normalize", False)

        self.cropper = Compose(cropper)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img = self.to_tensor(Image.open(self.samples[index]).convert("RGB"))

        if self.mode == utils.mode.TRAIN:
            if self.use_random_size:
                img = random_resize(img, target_size=self.gt_size)
            if self.use_rot:
                img = random_rotate(img)
            if self.use_flip:
                img = random_flip(img)

        img = self.cropper(img)

        if self.normalize:
            img = normalize(img, [0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        return img, img  # return identical image-pair

    def __str__(self):
        return self.__class__.__name__.lower()

    def __verbose__(self):
        logger = get_logger()
        logger.info(f"Create {self.__str__()} {self.mode} dataset with {self.__len__()} images")
