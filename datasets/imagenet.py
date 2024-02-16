from typing import AnyStr
from argparse import Namespace
import os

import utils.mode

from datasets.base.img_dataset import ImageDataset, make_dataset

DATA_ROOT = "data/imagenet"


def get_dataset(mode: AnyStr, args: Namespace):
    return ImageNet(mode, args, args.data_root.imagenet if args.get("data_root", False) else DATA_ROOT)


class ImageNet(ImageDataset):

    def __init__(self, mode: AnyStr, args: Namespace, data_root: AnyStr):

        if os.path.exists(f"{data_root}/train.txt") and os.path.exists(f"{data_root}/val.txt"):
            with open(f"{data_root}/{'train.txt' if mode == utils.mode.TRAIN else 'val.txt'}", 'r') as files:
                dataset = files.read().splitlines()
                dataset = dataset[:min(args.data_spec.get("max_datasize", float('inf')), len(dataset))]
        else:
            dataset = make_dataset(os.path.join(data_root, 'train' if mode == utils.mode.TRAIN else 'val'),
                                   max_dataset_size=args.data_spec.get("max_datasize", float("inf")))
        super(ImageNet, self).__init__(mode=mode, args=args, dataset=dataset)
        self.__verbose__()