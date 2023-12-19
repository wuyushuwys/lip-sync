import os

from glob import glob
from typing import Dict, AnyStr
from argparse import Namespace

import common.mode
from .video_dataset import FrameMelDataset

EXT = 'jpg'

DIR_PATH = lambda mode: f'data/HDTF/{mode}'


def get_dataset(mode: AnyStr, args: Namespace):
    return HDTF(mode, args)


class HDTF(FrameMelDataset):

    def __init__(self, mode: AnyStr, args: Namespace):
        folder_tree: Dict = dict()
        for folder in sorted(glob(f"{DIR_PATH(mode)}/*")):
            frame_list = sorted(filter(lambda x: x.endswith(EXT), os.listdir(folder)))
            if mode == common.mode.EVAL:
                frame_list = frame_list[:len(frame_list) // args.window_size * args.window_size]
            folder_tree[folder] = frame_list
        audio_cache_path = f"data/HDTF/HDTF_audio_sr_{args.audio_spec['sample_rate']}.h5"
        if not os.path.isfile(audio_cache_path):
            audio_cache_path = None
        super(HDTF, self).__init__(folder_tree=folder_tree, mode=mode, args=args, audio_cache_path=audio_cache_path)
