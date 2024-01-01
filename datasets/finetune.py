import os

from glob import glob
from typing import Dict, AnyStr
from argparse import Namespace

import utils.mode
from .video_dataset import FrameMelDataset

EXT = 'jpg'

# DIR_PATH = lambda mode: f'data/HDTF/{mode}'
META_PATH = lambda mode: f'data/finetune/{mode}.txt'


def get_dataset(mode: AnyStr, args: Namespace):
    return Finetune(mode, args)


class Finetune(FrameMelDataset):

    def __init__(self, mode: AnyStr, args: Namespace):
        folder_tree: Dict = dict()

        with open(META_PATH(mode), 'r') as f:
            lines = f.readlines()
            for line in lines:
                folder = line.strip('\n')
                folder_tree[folder] = sorted(filter(lambda x: x.endswith(EXT), os.listdir(folder)))
        # for folder in sorted(glob(f"{DIR_PATH(mode)}/*")):
        #     frame_list = sorted(filter(lambda x: x.endswith(EXT), os.listdir(folder)))
        #     if mode == utils.mode.EVAL:
        #         frame_list = frame_list[:len(frame_list) // args.window_size * args.window_size]
        #     folder_tree[folder] = frame_list
        audio_cache_path = f"data/finetune/finetune_audio_sr_{args.audio_spec['sample_rate']}.h5"
        if not os.path.isfile(audio_cache_path):
            audio_cache_path = None
        super(Finetune, self).__init__(folder_tree=folder_tree, mode=mode, args=args, audio_cache_path=audio_cache_path)
