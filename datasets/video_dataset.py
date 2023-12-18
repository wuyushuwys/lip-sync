import os
import random

from pathlib import Path
from typing import Dict
from argparse import Namespace

import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms.functional import resize, InterpolationMode
from torchvision.io import read_image

import common
from utils.audio import load_wav, melspectrogram
from utils.logging_tool import get_logger
import time

class FrameMelDataset(Dataset):

    def __init__(self, folder_tree: Dict, mode: str, args: Namespace):
        super().__init__()

        # dataset tree
        # {"folder_path_per_video": [img1, img2, ...]}
        # root
        #   ├── vid1
        #   │    ├── xxxxx.image : image file
        #   │    ├── ......
        #   │    └── audio.wav (skip) : obtained afterwards
        #   ├── ......
        #   └── ......

        self.folder_tree = folder_tree
        self.root_key = list(folder_tree.keys())  # list of each video data folder
        self.num_samples = args.num_samples  # number of samples from each video
        self.video_spec = args.video_spec
        self.audio_spec = args.audio_spec
        self.window_size = args.window_size
        self.model = args.model
        self.mode = mode
        # Indexing eval list
        if self.mode == common.mode.EVAL:
            eval_filelist = []
            eval_length = {}
            for folder, v in folder_tree.items():
                eval_frames = sorted(map(lambda fname: os.path.join(folder, fname), v))
                eval_filelist.extend(eval_frames)
                eval_length[folder] = len(eval_frames)
            self.eval_filelist = eval_filelist
            self.eval_length = eval_length
        logger = get_logger(args.job_dir)
        logger.info(f"Load {len(folder_tree)} video data in {mode}")
        self.al_time = 0

    def __len__(self):
        if self.mode == common.mode.TRAIN:
            return len(self.folder_tree) * self.num_samples
        else:
            return len(self.eval_filelist) // self.window_size - 1

    def __getitem__(self, index):
        if self.model == 'syncnet':
            if self.mode == common.mode.TRAIN:
                frame_list, audio_file = self._load_index(index)
                img_window, mel, label = self._load_sync_train_data(frame_list, audio_file)
                return img_window, mel, label
            else:
                self.index = index
                frame_window = self.eval_filelist[index * self.window_size: (index + 1) * self.window_size]
                assert len(frame_window) == self.window_size
                audio_file = Path(frame_window[0]).parent / 'audio.wav'
                img_window, mel, label = self._load_sync_eval_data(frame_window, audio_file)
                return img_window, mel, label
        else:
            NotImplementedError()

    def _load_index(self, item):
        index_folder = self.root_key[item // self.num_samples]
        frame_list = self.folder_tree[index_folder]
        audio_file = Path(self.folder_tree[index_folder]) / 'audio.wav'
        return frame_list, audio_file

    def _load_sync_train_data(self, frame_list, audio_file):
        idx, false_idx = random.sample(range(len(frame_list) - self.window_size), 2)

        img_window = self._load_frame_window(frame_list, idx)

        return self._load_colorsync_data(idx, false_idx, img_window, audio_file)

    def _load_frame_window(self, fname_list, index):
        window = []
        for fname in fname_list[index:index + self.window_size]:
            img = resize(read_image(fname), eval(self.video_spec['size']),
                         interpolation=InterpolationMode.BILINEAR,
                         antialias=True)
            window.append(img)
        return window

    def _load_sync_eval_data(self, frame_window, audio_file):
        idx = eval(Path(frame_window[0]).stem.lstrip('0')) - 1  # int

        # same video check
        for fname in frame_window[1:]:
            assert Path(frame_window[0]).parent == Path(fname).parent

        video_length = self.eval_length[os.path.split(frame_window[0])[0]]
        false_idx = random.choice([i for i in range(video_length - self.window_size) if i != idx])

        img_window = self._load_frame_window_fname(frame_window)

        return self._load_colorsync_data(idx, false_idx, img_window, audio_file)

    def _load_colorsync_data(self, idx, false_idx, img_window, audio_file):

        if random.random() > 0.5:
            audio_idx = idx
            label = torch.ones(1)
        else:
            audio_idx = false_idx
            label = torch.zeros(1)

        st = time.time()
        mel = self._load_audio_melspec(audio_file)
        print(time.time() - st)
        mel = self._crop_audio_window(mel.copy(), audio_idx)
        assert mel.shape[0] == self.audio_spec['mel_step_size'], f"{mel.shape[0]} {self.audio_spec['mel_step_size']}"

        img_window = torch.cat(img_window, dim=0) / 255 - 0.5
        img_window = img_window[:, img_window.size(1) // 2:, :]

        mel = torch.tensor(mel.T, dtype=torch.float).unsqueeze(0)

        return img_window, mel, label

    def _load_frame_window_fname(self, fname_list):
        window = []
        for fname in fname_list:
            img = resize(read_image(fname), self.video_spec['size'],
                         interpolation=InterpolationMode.BILINEAR,
                         antialias=True)
            window.append(img)
        return window

    def _load_audio_melspec(self, file_name):
        wav = load_wav(path=file_name, sr=self.audio_spec['sample_rate'])
        return melspectrogram(wav).T

    def _crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_idx = int(80. * (start_frame / self.video_spec['fps']))  # 80 = 16000 / 200

        end_idx = start_idx + self.audio_spec['mel_step_size']

        return spec[start_idx: end_idx, :]
