from typing import AnyStr
from argparse import Namespace

import utils.mode

from datasets.base.img_dataset import ImageDataset, make_dataset

DATA_ROOT = "data/CFD"


def get_dataset(mode: AnyStr, args: Namespace):
    return CFD(mode, args, args.data_root.cfd if args.get("data_root", False) else DATA_ROOT)


class CFD(ImageDataset):

    def __init__(self, mode: AnyStr, args: Namespace, data_root: AnyStr):

        dataset = make_dataset(dir=data_root)
        num_eval = min(int(args.data_spec.num_eval), int(len(dataset) * 0.025))
        if mode == utils.mode.TRAIN:
            dataset = dataset[:-num_eval]
        else:
            dataset = dataset[-num_eval:]
        super(CFD, self).__init__(mode=mode, args=args, dataset=dataset)
        self.__verbose__()
