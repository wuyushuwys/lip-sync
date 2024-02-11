from typing import AnyStr
from argparse import Namespace
import os

import utils.mode

from .img_dataset import ImageDataset, make_dataset

DATA_ROOT = "data/imagenet"


def get_dataset(mode: AnyStr, args: Namespace, data_root: AnyStr = DATA_ROOT):
    return ImageNet(mode, args, data_root)


class ImageNet(ImageDataset):

    def __init__(self, mode: AnyStr, args: Namespace, data_root: AnyStr):
        if mode == utils.mode.TRAIN:
            subset = 'train'
        else:
            subset = 'val'
        dataset = make_dataset(dir=os.path.join(data_root, subset))
        super(ImageNet, self).__init__(mode=mode, args=args, dataset=dataset)
        self.__verbose__()