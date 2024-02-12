from typing import AnyStr
from argparse import Namespace

import utils.mode

from datasets.base.img_dataset import ImageDataset, make_dataset

DATA_ROOT = "data/CelebA-HQ"


def get_dataset(mode: AnyStr, args: Namespace):
    return CelebA_HQ(mode, args, args.data_root.celeba if args.get("data_root", False) else DATA_ROOT)


class CelebA_HQ(ImageDataset):

    def __init__(self, mode: AnyStr, args: Namespace, data_root: AnyStr):

        dataset = make_dataset(dir=data_root, max_dataset_size=args.data_spec.get('max_datasize', float('inf')))
        num_eval = min(int(args.data_spec.num_eval), int(len(dataset) * 0.025))
        if mode == utils.mode.TRAIN:
            dataset = dataset[:-num_eval]
        else:
            dataset = dataset[-num_eval:]
        super(CelebA_HQ, self).__init__(mode=mode, args=args, dataset=dataset)
        self.__verbose__()
