import os

from glob import glob
from typing import Dict, AnyStr
from argparse import Namespace

from common.io import Hdf5

from .video_dataset import FrameMelDataset
from .utils import load_from_folder

DATA_ROOT = 'data/LRS2'
META_PATH = lambda mode: f'{DATA_ROOT}/{mode}.txt'


def get_dataset(mode: AnyStr, args: Namespace):
    return LRS2(mode, args)


class LRS2(FrameMelDataset):
    EXT = 'jpg'

    def __init__(self, mode: AnyStr, args: Namespace):
        folder_tree: Dict = dict()

        with open(META_PATH(mode), 'r') as f:
            lines = f.readlines()
            for line in lines:
                folder = os.path.join(f'{DATA_ROOT}/data', line.strip('\n'))
                load_from_folder(folder_tree=folder_tree, folder=folder, mode=args.data_spec['mode'], ext=self.EXT)

        audio_cache_path = f"{DATA_ROOT}/LRS2_audio_sr_{args.audio_spec['sample_rate']}.h5"
        if not os.path.isfile(audio_cache_path):
            audio_cache_path = None
        super(LRS2, self).__init__(folder_tree=folder_tree, mode=mode, args=args, data_mode=args.data_spec['mode'],
                                   audio_cache_path=audio_cache_path)

    def __str__(self):
        return 'LRS2'