from typing import AnyStr
from argparse import Namespace

import utils.mode

from .img_dataset import ImageDataset

DATA_ROOT = "data/open_images"


def get_dataset(mode: AnyStr, args: Namespace, data_root: AnyStr = DATA_ROOT):
    return OpenImages(mode, args, data_root)


class OpenImages(ImageDataset):

    def __init__(self, mode: AnyStr, args: Namespace, data_root: AnyStr):
        self.mode = mode
        with open(f"{data_root}/{'train.txt' if mode == utils.mode.TRAIN else 'validation.txt'}", 'r') as files:
            dataset = files.read().splitlines()[:args.data_spec.max_datasize]

        super(OpenImages, self).__init__(mode=mode, args=args, dataset=dataset)

        self.__verbose__()
