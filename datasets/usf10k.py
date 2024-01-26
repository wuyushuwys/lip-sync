from typing import AnyStr
from argparse import Namespace

import torchvision

torchvision.disable_beta_transforms_warning()


from .img_dataset import ImageDataset

DATA_ROOT = "data/usf10k"


def get_dataset(mode: AnyStr, args: Namespace, data_root: AnyStr = DATA_ROOT):
    return USF10K(mode, args, data_root)


class USF10K(ImageDataset):

    def __init__(self, mode: AnyStr, args: Namespace, data_root: AnyStr):
        super(USF10K, self).__init__(mode=mode, args=args, data_root=data_root)
