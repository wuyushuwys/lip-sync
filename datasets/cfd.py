from typing import AnyStr
from argparse import Namespace
from .img_dataset import ImageDataset

DATA_ROOT = "data/CFD"


def get_dataset(mode: AnyStr, args: Namespace, data_root: AnyStr = DATA_ROOT):
    return CFD(mode, args, data_root)


class CFD(ImageDataset):

    def __init__(self, mode: AnyStr, args: Namespace, data_root: AnyStr):
        super(CFD, self).__init__(mode=mode, args=args, data_root=data_root)
