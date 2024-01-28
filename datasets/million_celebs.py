from typing import AnyStr
from argparse import Namespace
from .img_dataset import ImageDataset

DATA_ROOT = "data/MillionCelebs"


def get_dataset(mode: AnyStr, args: Namespace, data_root: AnyStr = DATA_ROOT):
    return MillionCelebs(mode, args, data_root)


class MillionCelebs(ImageDataset):

    def __init__(self, mode: AnyStr, args: Namespace, data_root: AnyStr):
        super(MillionCelebs, self).__init__(mode=mode, args=args, data_root=data_root)
        self.__verbose__()