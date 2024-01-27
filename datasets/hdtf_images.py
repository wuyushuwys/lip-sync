from typing import AnyStr
from argparse import Namespace

import torchvision

torchvision.disable_beta_transforms_warning()


from .img_dataset import ImageDataset

DATA_ROOT = "data/HDTF_Image"


def get_dataset(mode: AnyStr, args: Namespace, data_root: AnyStr = DATA_ROOT):
    return HDTF_Image(mode, args, data_root)


class HDTF_Image(ImageDataset):

    def __init__(self, mode: AnyStr, args: Namespace, data_root: AnyStr):
        super(HDTF_Image, self).__init__(mode=mode, args=args, data_root=data_root, sample_rate=25)

