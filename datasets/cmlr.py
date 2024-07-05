import os

from typing import Dict, AnyStr
from argparse import Namespace

from datasets.base.video_dataset import FrameMelDataset
from .utils import load_from_folder

DATA_ROOT = 'data/CMLR'
META_PATH = lambda data_root, mode: f'{data_root}/{mode}.txt'


def get_dataset(mode: AnyStr, args: Namespace):
    return CMLR(mode, args, args.data_root.lrs2 if args.get("data_root", False) else DATA_ROOT)


class CMLR(FrameMelDataset):
    EXT = 'jpg'

    def __init__(self, mode: AnyStr, args: Namespace, data_root: AnyStr):
        folder_tree: Dict = dict()

        with open(META_PATH(data_root, mode), 'r') as f:
            lines = f.readlines()
            for line in lines:
                folder = os.path.join(f'{data_root}/data', line.strip('\n'))
                load_from_folder(folder_tree=folder_tree, folder=folder, mode=args.data_spec['mode'], ext=self.EXT)

        video_cache_path = f"{data_root}/data.h5" if os.path.isfile(f"{data_root}/data.h5") else None
        audio_cache_path = f"{data_root}/CMLR_audio_sr_{args.audio_spec['sample_rate']}.h5"
        if not os.path.isfile(audio_cache_path):
            audio_cache_path = None
        super(CMLR, self).__init__(folder_tree=folder_tree, mode=mode, args=args, data_mode=args.data_spec['mode'],
                                   audio_cache_path=audio_cache_path, video_cache_path=video_cache_path)

    def __str__(self):
        return self.__class__.__name__.lower()
