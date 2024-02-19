import os

from typing import AnyStr
from argparse import Namespace

import utils.mode

from datasets.base.img_dataset import ImageDataset, make_dataset

DATA_ROOT = "data/HDTF_Image_original"


def get_dataset(mode: AnyStr, args: Namespace):
    return HDTFImagesOrg(mode, args, args.data_root.hdtf_images if args.get("data_root", False) else DATA_ROOT)


class HDTFImagesOrg(ImageDataset):

    def __init__(self, mode: AnyStr, args: Namespace, data_root: AnyStr):
        sample_rate = 25 * 180
        if os.path.exists(f"{data_root}/filelist.txt"):
            with open(f"{data_root}/filelist.txt", 'r') as files:
                dataset = files.read().splitlines()
        else:
            dataset = make_dataset(dir=data_root)
        num_eval = min(int(args.data_spec.num_eval), int(len(dataset) * 0.025))
        if mode == utils.mode.TRAIN:
            dataset = dataset[:-num_eval][::sample_rate]
        else:
            dataset = dataset[-num_eval:][::sample_rate]

        super(HDTFImagesOrg, self).__init__(mode=mode, args=args, dataset=dataset)
        self.__verbose__()
