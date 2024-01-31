from typing import AnyStr
from argparse import Namespace

import utils.mode

from .img_dataset import ImageDataset, make_dataset

DATA_ROOT = "data/usf10k"


def get_dataset(mode: AnyStr, args: Namespace, data_root: AnyStr = DATA_ROOT):
    return USF10K(mode, args, data_root)


class USF10K(ImageDataset):

    def __init__(self, mode: AnyStr, args: Namespace, data_root: AnyStr):

        dataset = make_dataset(dir=data_root)
        num_eval = min(int(args.data_spec.num_eval), int(len(dataset) * 0.025))
        if mode == utils.mode.TRAIN:
            dataset = dataset[:-num_eval]
        else:
            dataset = dataset[-num_eval:]
        super(USF10K, self).__init__(mode=mode, args=args, dataset=dataset)
        self.__verbose__()
