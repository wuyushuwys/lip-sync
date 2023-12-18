import os

from glob import glob
from typing import Dict
from argparse import Namespace

import common.mode
from .video_dataset import FrameMelDataset

EXT = 'jpg'

DIR_PATH = lambda mode: f'data/HDTF/{mode}'


class HDTF(FrameMelDataset):

    def __init__(self, mode: str, args: Namespace):
        folder_tree: Dict = dict()
        for folder in sorted(glob(f"{DIR_PATH(mode)}/*")):
            frame_list = sorted(filter(lambda x: x.endswith(EXT), os.listdir(folder)))
            if mode == common.mode.EVAL:
                frame_list = frame_list[:len(frame_list) // args.window_size * args.window_size]
            folder_tree[folder] = frame_list

        super(HDTF, self).__init__(folder_tree=folder_tree, mode=mode, args=args)
