import os

from glob import glob
from typing import Dict, AnyStr
from argparse import Namespace

from .video_dataset import FrameMelDataset
from .utils import load_from_folder

DATA_ROOT = "data/TH1K"
META_PATH = lambda mode: f'{DATA_ROOT}/{mode}.txt'


def get_dataset(mode: AnyStr, args: Namespace):
    return TH1K(mode, args)


class TH1K(FrameMelDataset):
    EXT = 'jpg'

    def __init__(self, mode: AnyStr, args: Namespace):
        folder_tree: Dict = dict()

        folder_list = os.listdir(os.path.join(DATA_ROOT, mode))

        for folder in folder_list:
            load_from_folder(folder_tree=folder_tree, folder=folder, mode=args.data_spec['mode'], ext=self.EXT)

        audio_cache_path = f"{DATA_ROOT}/TH1K_audio_sr_{args.audio_spec['sample_rate']}.h5"
        if not os.path.isfile(audio_cache_path):
            audio_cache_path = None
        super(TH1K, self).__init__(folder_tree=folder_tree, mode=mode, args=args, data_mode=args.data_spec['mode'],
                                   audio_cache_path=audio_cache_path)
