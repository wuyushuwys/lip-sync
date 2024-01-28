from typing import AnyStr
from argparse import Namespace
from .img_dataset import ImageDataset

DATA_ROOT = "data/lfw"


def get_dataset(mode: AnyStr, args: Namespace, data_root: AnyStr = DATA_ROOT):
    return LFW(mode, args, data_root)


class LFW(ImageDataset):

    def __init__(self, mode: AnyStr, args: Namespace, data_root: AnyStr):
        super(LFW, self).__init__(mode=mode, args=args, data_root=data_root)
        self.__verbose__()