from typing import AnyStr
from argparse import Namespace
from .ffhq import FFHQ

DATA_ROOT = "data/CelebA-HQ"


def get_dataset(mode: AnyStr, args: Namespace, data_root: AnyStr = DATA_ROOT):
    return CelebA_HQ(mode, args, data_root)


class CelebA_HQ(FFHQ):

    def __init__(self, mode: AnyStr, args: Namespace, data_root: AnyStr):
        super(CelebA_HQ, self).__init__(mode=mode, args=args, data_root=data_root)
