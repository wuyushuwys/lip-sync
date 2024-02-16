import math
import os
import random

from pathlib import Path
from typing import Dict, AnyStr
from argparse import Namespace
from datetime import timedelta
from tqdm import tqdm

import numpy as np

import torch
import torchvision

torchvision.disable_beta_transforms_warning()
from torch.utils.data.dataset import Dataset
from torchvision.transforms.functional import resize, InterpolationMode
from torchvision.transforms.v2 import (Compose, Normalize,
                                       RandomRotation, RandomHorizontalFlip)
from torchvision.io import read_image

import common
import utils
from utils.audio import load_wav, melspectrogram
from utils.logging_tool import get_logger
from utils.init_utils import get_dist_info


class FrameMelDataset(Dataset):

    def __init__(self, folder_tree: Dict, mode: AnyStr, args: Namespace,
                 data_mode: AnyStr = 'image',
                 audio_cache_path: AnyStr = None,
                 video_cache_path: AnyStr = None,
                 skip_offset: float = None,
                 bottom_half: bool = True) -> None:
        super().__init__()

        logger = get_logger()

        # dataset tree
        # {"folder_path_per_video": [img1, img2, ...]}
        # path
        #   ├── vid1
        #   │    ├── xxxxx.image : image file
        #   │    ├── ......
        #   │    └── audio.wav (skip) : obtained afterwards
        #   ├── ......
        #   └── ......

        self.folder_tree = folder_tree
        self.root_key = list(folder_tree.keys())  # list of each video data folder
        self.video_spec = args.video_spec
        self.audio_spec = args.audio_spec
        self.data_spec = args.data_spec
        self.window_size = args.data_spec['window_size']
        self.arch = args.arch
        self.mode = mode
        self.data_mode = data_mode
        self.skip_offset = skip_offset
        self.bottom_half = bottom_half
        if self.mode == utils.mode.TRAIN:
            self.num_samples = self.data_spec.num_samples  # number of samples from each video
        elif self.mode == utils.mode.EVAL:
            self.num_samples = self.data_spec.eval_samples  # number of samples from each video

        # if self.model == 'syncnet':
        #     # Indexing eval list
        #     if self.mode == utils.mode.EVAL:
        #         eval_filelist = []
        #         eval_length = {}
        #         for folder, v in folder_tree.items():
        #             if self.data_mode == 'image':
        #                 eval_frames = sorted(map(lambda k: os.path.join(folder, k), v))
        #             elif self.data_mode == 'h5':
        #                 eval_frames = sorted(map(lambda k: os.path.join(folder, k), v.keys))
        #             else:
        #                 raise NotImplementedError(f"{self.data_mode} not supported")
        #             eval_filelist.extend(eval_frames[:len(eval_frames) // self.window_size * self.window_size])
        #             eval_length[folder] = len(eval_frames)
        #
        #         self.eval_filelist = eval_filelist
        #         self.eval_length = eval_length
        if video_cache_path:
            self.video_cache = common.io.Hdf5(video_cache_path)
            logger.info(f"{self}: Loading video cache: {video_cache_path}")
        else:
            self.video_cache = None
            if 'verbose' in self.data_spec.keys() and self.data_spec['verbose']:
                rank, _ = get_dist_info()
                iterator = tqdm(folder_tree.values(), dynamic_ncols=True) if rank == 0 else folder_tree.values()
                if self.data_mode == 'image':
                    load_frames = sum(len(v) for v in iterator)
                elif self.data_mode == 'h5':
                    load_frames = sum(len(v.keys) for v in iterator)
                else:
                    raise NotImplementedError(f"{self.data_mode} not supported")

                logger.info(f"total video length approx. {timedelta(seconds=load_frames // self.video_spec['fps'])}")
        logger.info(f"{self}: Load {len(folder_tree)} video clips in {mode}")

        if audio_cache_path:
            self.audio_cache = common.io.Hdf5(audio_cache_path)
            logger.info(f"{self}: Loading audio cache: {audio_cache_path}")
        else:
            self.audio_cache = None
            logger.info(f"{self}: Loading audio from file")

        transforms = []
        if self.data_spec.get('aug', False):
            aug_spec = self.data_spec.aug
            if aug_spec.get('rotate', False):
                transforms.append(RandomRotation(**aug_spec.rotate,
                                                 interpolation=InterpolationMode.BILINEAR,
                                                 fill=1))
            if aug_spec.get('flip', False):
                transforms.append(RandomHorizontalFlip(**aug_spec.flip))
        if self.data_spec.get('normalize', False):
            transforms.append(Normalize(**self.data_spec.normalize))
        self.transform = Compose(transforms)

    def __len__(self):
        # if self.model == 'syncnet' and self.mode == utils.mode.EVAL:
        #     return len(self.eval_filelist) // self.window_size - 1
        # else:
        return len(self.folder_tree) * self.num_samples

    def __getitem__(self, index):
        if self.arch == 'syncnet':
            # if self.mode == utils.mode.TRAIN:
            frame_list, audio_file = self._load_index(index)
            img_window, mel, label = self._load_sync_train_data(frame_list, audio_file)
            return img_window, mel, label
            # else:
            #     frame_window = self.eval_filelist[index * self.window_size: (index + 1) * self.window_size]
            #     assert len(frame_window) == self.window_size
            #     audio_file = Path(frame_window[0]).parent / 'audio.wav'
            #     img_window, mel, label = self._load_sync_eval_data(frame_window, audio_file)
            #     return img_window, mel, label
        elif self.arch in ['lipsync', 'mage']:
            """
            x: (ch, T, H, W)
            indiv_mels: (1, T, 80, mel_step_size)
            mel: (1, 80, mel_step_size)
            y: (ch, T, H, W)
            """
            frame_list, audio_file = self._load_index(index)
            x, indiv_mels, mel, y = self._load_lipsync_train_data(frame_list, audio_file)
            if self.data_spec.get('singular', False):
                return x[:, 2], indiv_mels[2], mel, y[:, 2]
            return x, indiv_mels, mel, y
        else:
            raise NotImplementedError(f"{self.arch} is not implemented")

    def _load_index(self, item):
        index_folder = self.root_key[item // self.num_samples]
        frame_list = self.folder_tree[index_folder]
        if self.data_mode == 'image':
            frame_list = [os.path.join(index_folder, fname) for fname in frame_list]
        elif self.data_mode == 'h5':
            if self.video_cache:
                frame_list = [os.path.join(index_folder, fname) for fname in self.video_cache.iter_keys(index_folder)]
            else:
                frame_list = [os.path.join(index_folder, fname) for fname in frame_list.keys]
        else:
            raise NotImplementedError(f"{self.data_mode}")
        audio_file = Path(index_folder) / 'audio.wav'
        return frame_list, audio_file

    def _load_image(self, fname):
        if self.data_mode == 'image':
            return read_image(fname)
        elif self.data_mode == 'h5':
            if self.video_cache:
                return torch.from_numpy(np.ascontiguousarray(self.video_cache.get(fname)))
            else:
                root, key = os.path.split(fname)
                return torch.from_numpy(np.ascontiguousarray(self.folder_tree[root].get(key)))
        else:
            raise NotImplementedError()

    def _load_frame_window_fname(self, fname_list):
        window = []
        for fname in fname_list:
            img = resize(self._load_image(fname), self.video_spec['size'],
                         interpolation=InterpolationMode.BILINEAR,
                         antialias=True)
            window.append(img)
        return window

    def _load_frame_window(self, fname_list, index):
        window = []
        for fname in fname_list[index:index + self.window_size]:
            img = resize(self._load_image(fname), self.video_spec['size'],
                         interpolation=InterpolationMode.BILINEAR,
                         antialias=True)
            window.append(img)
        return window

    def _load_lipsync_train_data(self, frame_list, audio_file):
        false_offset = random.randint(self.video_spec.fps, self.video_spec.fps * 2)  # range for false frame
        if self.skip_offset:
            skip_start = 2 + int(self.skip_offset * len(frame_list))
            skip_end = int(len(frame_list) - self.skip_offset * len(frame_list) - self.window_size) - 3
            if skip_end - skip_start < self.window_size + 1:
                skip_start = 2
                skip_end = len(frame_list) - self.window_size - 3
            # idx = random.sample(range(skip_start, skip_end), 1)
            # idx = random.choice(range(skip_start, skip_end))
            # if idx + false_offset < len(frame_list) - self.window_size:
            idx, false_idx = random.sample(range(skip_start, skip_end), 2)
            # else:
            #     false_idx = idx + false_offset

        else:
            skip_start = 2
            skip_end = len(frame_list) - self.window_size - 3
            # idx = random.sample(range(skip_start, skip_end), 1)
            # idx = random.choice(range(skip_start, skip_end))
            # if idx + false_offset < len(frame_list) - self.window_size:
            idx, false_idx = random.sample(range(skip_start, skip_end), 2)
            # else:
            #     false_idx = idx + false_offset

        true_window = self._load_frame_window(frame_list, idx)
        wrong_window = self._load_frame_window(frame_list, false_idx)
        data = self._load_lipsync_data(idx, true_window=true_window, wrong_window=wrong_window,
                                       audio_file=audio_file)

        return data

    def _load_lipsync_data(self, idx, true_window, wrong_window, audio_file):
        assert idx - 2 >= 0
        audio_mel = self._load_audio_melspec(audio_file)
        mel = self._crop_audio_window(audio_mel.copy(), idx)
        indiv_mels = self._segmented_mels(audio_mel.copy(), idx)

        true_window = torch.stack(true_window, dim=1) / 255
        wrong_window = torch.stack(wrong_window, dim=1) / 255

        if hasattr(self, 'transform'):
            # if self.mode == utils.mode.TRAIN:
            true_window = true_window.permute(1, 0, 2, 3)
            wrong_window = wrong_window.permute(1, 0, 2, 3)
            window = torch.cat([true_window, wrong_window], dim=0)
            window = self.transform(window)
            true_window, wrong_window = window.split(self.window_size, dim=0)
            # true_window, wrong_window = self.transform([true_window, wrong_window])
            true_window = true_window.permute(1, 0, 2, 3)
            wrong_window = wrong_window.permute(1, 0, 2, 3)

        gt = true_window.clone()

        # if 'crop_pad' in self.video_spec.keys():
        #     if self.mode == utils.mode.TRAIN:
        #
        #         if exists('random_crop', self.video_spec) and self.video_spec['random_crop']:
        #             y1, y2, x1, x2 = [random.randint(b // 2, b) if b > 0 else random.randint(b, b // 2) for b in
        #                               self.video_spec['crop_pad']]
        #         else:
        #             y1, y2, x1, x2 = self.video_spec['crop_pad']
        #         true_window[:, :, true_window.size(2) // 2 + y1: y2, x1: x2] = 0
        #     else:
        #         y1, y2, x1, x2 = self.video_spec['crop_pad']
        #         true_window[:, :, true_window.size(2) // 2 + y1: y2, x1: x2] = 0
        # else:
        #     true_window[:, :, true_window.size(2) // 2:] = 0

        x = torch.cat([true_window, wrong_window], dim=0)
        indiv_mels = torch.tensor(indiv_mels, dtype=torch.float).unsqueeze(1)
        mel = torch.tensor(mel.T, dtype=torch.float).unsqueeze(0)
        y = gt

        return x, indiv_mels, mel, y

    def _load_sync_train_data(self, frame_list, audio_file):
        if self.skip_offset:
            skip_start = int(self.skip_offset * len(frame_list))
            skip_end = int(len(frame_list) - self.skip_offset * len(frame_list) - self.window_size - 1)
            if skip_end - skip_start < self.window_size + 1:
                skip_start = 0
                skip_end = len(frame_list) - self.window_size - 1
            idx, false_idx = random.sample(range(skip_start, skip_end), 2)
        else:
            idx, false_idx = random.sample(range(len(frame_list) - self.window_size - 1), 2)
        img_window = self._load_frame_window(frame_list, idx)

        return self._load_colorsync_data(idx, false_idx, img_window, audio_file)

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

        mel = self._load_audio_melspec(audio_file)
        mel = self._crop_audio_window(mel.copy(), audio_idx)

        if 'aug' in self.data_spec.keys() and self.data_spec['aug']:
            if self.mode == utils.mode.TRAIN:
                img_window = torch.stack(img_window, dim=0) / 255
                img_window = self.transform(img_window)
                if self.bottom_half:
                    img_window = img_window[..., img_window.size(2) // 2:, :].contiguous()
                t, c, h, w = img_window.size()
                img_window = img_window.reshape(t * c, h, w)
            else:
                img_window = torch.cat(img_window, dim=0) / 255
                if self.bottom_half:
                    img_window = img_window[..., img_window.size(1) // 2:, :].contiguous()
        else:
            img_window = torch.cat(img_window, dim=0) / 255
            if self.bottom_half:
                img_window = img_window[..., img_window.size(1) // 2:, :].contiguous()

        mel = torch.tensor(mel.T, dtype=torch.float).unsqueeze(0)

        return img_window, mel, label

    @staticmethod
    def _aug_mask_mel(crop_mel):
        block_size = 0.1
        time_size = math.ceil(block_size * crop_mel.shape[0])
        freq_size = math.ceil(block_size * crop_mel.shape[1])
        time_lim = crop_mel.shape[0] - time_size
        freq_lim = crop_mel.shape[1] - freq_size

        time_st = random.randint(0, time_lim)
        freq_st = random.randint(0, freq_lim)

        mel = crop_mel.copy()
        mel[time_st:time_st + time_size] = -4.
        mel[:, freq_st:freq_st + freq_size] = -4.

        return mel

    def _load_audio_melspec(self, file_name):
        if not self.audio_cache:
            npy_name = os.path.splitext(file_name)[0] + '.npy'
            if not os.path.isfile(npy_name):
                wav = load_wav(path=file_name, sr=self.audio_spec['sample_rate'])
                mel = melspectrogram(wav).T
            else:
                mel = np.load(npy_name)
        else:
            mel = np.asarray(self.audio_cache.get(str(file_name)))

        return mel

    def _crop_audio_window(self, spec, start_frame_num):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_idx = int(80. * (start_frame_num / self.video_spec['fps']))  # 80 = 16000 / 200

        end_idx = start_idx + self.audio_spec['mel_step_size']
        if end_idx > spec.shape[0]:
            # adjust window avoid error (may introduce some not perfect matched data. but won't raise error)
            start_idx = end_idx - self.audio_spec['mel_step_size']
            end_idx = spec.shape[0]
        mel = spec[start_idx: end_idx, :]
        # if self.mode == utils.mode.TRAIN and random.random() < 0.3:
        #     mel = self._aug_mask_mel(mel)
        assert mel.shape[0] == self.audio_spec[
            'mel_step_size'], f"{start_frame_num} {start_idx} {mel.shape[0]} {self.audio_spec['mel_step_size']} {spec.shape}"
        return mel

    def _segmented_mels(self, spec, start_frame_num):
        mels = []
        assert start_frame_num - 2 >= 0
        for i in range(start_frame_num, start_frame_num + self.window_size):
            try:
                mels.append(self._crop_audio_window(spec, i - 2).T)
            except AssertionError:
                raise AssertionError(f"{start_frame_num} {i} {start_frame_num + self.window_size}")
        return np.asarray(mels)
