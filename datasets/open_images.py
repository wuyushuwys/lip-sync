import random
import os
from pathlib import Path
from glob import glob
from typing import AnyStr
from argparse import Namespace
from torchvision.datasets.folder import is_image_file

from .img_dataset import ImageDataset

DATA_ROOT = "data/open_images"


def get_dataset(mode: AnyStr, args: Namespace, data_root: AnyStr = DATA_ROOT):
    return OpenImages(mode, args, data_root)


class OpenImages(ImageDataset):

    def __init__(self, mode: AnyStr, args: Namespace, data_root: AnyStr):

        super(OpenImages, self).__init__(mode=mode, args=args, data_root=data_root, trace_data=False)
        self.mode = mode
        data_root = f"{DATA_ROOT}/{'train' if mode == 'train' else 'test'}/**/*"
        self.samples = list(filter(lambda fname: is_image_file(fname), glob(data_root, recursive=True)))
        self.__verbose__()
