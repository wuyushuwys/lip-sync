import os.path
from typing import AnyStr
from argparse import Namespace

import utils.mode

from datasets.base.img_dataset import ImageDataset
from datasets.base.img_dataset import make_dataset

DATA_ROOT = "data/open_images"


def get_dataset(mode: AnyStr, args: Namespace):
    return OpenImages(mode, args, args.data_root.open_images if args.get("data_root", False) else DATA_ROOT)


class OpenImages(ImageDataset):

    def __init__(self, mode: AnyStr, args: Namespace, data_root: AnyStr):
        self.mode = mode
        if os.path.exists(f"{data_root}/train.txt") and os.path.exists(f"{data_root}/validation.txt"):
            with open(f"{data_root}/{'train.txt' if mode == utils.mode.TRAIN else 'validation.txt'}", 'r') as files:
                dataset = files.read().splitlines()
                dataset = dataset[:min(args.data_spec.get("max_datasize", float('inf'), len(dataset)))]
        else:
            dataset = make_dataset(os.path.join(data_root, 'train' if mode == utils.mode.TRAIN else 'validation'),
                                   max_dataset_size=args.data_spec.get("max_datasize", float("inf")))

        super(OpenImages, self).__init__(mode=mode, args=args, dataset=dataset)

        self.__verbose__()
