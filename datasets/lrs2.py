import os

from glob import glob
from typing import Dict, AnyStr
from argparse import Namespace

from common.io import Hdf5
from .video_dataset import FrameMelDataset

# DIR_PATH = lambda mode: f'data/HDTF/{mode}'
META_PATH = lambda mode: f'data/LRS2/{mode}.txt'


def get_dataset(mode: AnyStr, args: Namespace):
    return LRS2(mode, args)


class LRS2(FrameMelDataset):
    EXT = 'jpg'

    def __init__(self, mode: AnyStr, args: Namespace):
        folder_tree: Dict = dict()

        with open(META_PATH(mode), 'r') as f:
            lines = f.readlines()
            for line in lines:
                folder = os.path.join('data/LRS2', line.strip('\n'))
                if args.data_mode == 'image':
                    folder_tree[folder] = sorted(filter(lambda x: x.endswith(self.EXT), os.listdir(folder)))
                elif args.data_mode == 'h5':
                    folder_tree[folder] = Hdf5(os.path.join(folder, 'cache.h5'))
                else:
                    raise NotImplementedError(f"{args.data_mode} not supported")
        # for folder in sorted(glob(f"{DIR_PATH(mode)}/*")):
        #     frame_list = sorted(filter(lambda x: x.endswith(EXT), os.listdir(folder)))
        #     if mode == utils.mode.EVAL:
        #         frame_list = frame_list[:len(frame_list) // args.window_size * args.window_size]
        #     folder_tree[folder] = frame_list
        audio_cache_path = f"data/LRS2/LRS2_audio_sr_{args.audio_spec['sample_rate']}.h5"
        if not os.path.isfile(audio_cache_path):
            audio_cache_path = None
        super(LRS2, self).__init__(folder_tree=folder_tree, mode=mode, args=args, data_mode=args.data_mode,
                                   audio_cache_path=audio_cache_path)
