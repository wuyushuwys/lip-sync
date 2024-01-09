import copy
import cv2
import math
import os
from itertools import cycle

from typing import Dict, AnyStr
from pathlib import Path
from glob import glob
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import resize, InterpolationMode


class ImageFolder(Dataset):

    def __init__(self, folder, ext='jpg', output_mode='tensor') -> None:
        super().__init__()
        self.img_lists = sorted(glob(f"{folder}/*.{ext}"))
        self.output_mode = output_mode

    def __len__(self):
        return len(self.img_lists)

    def __getitem__(self, index: int) -> Dict:
        fname = self.img_lists[index]
        if self.output_mode == 'tensor':
            img = read_image(fname)
        elif self.output_mode == 'cv2':
            img = cv2.imread(fname)
        elif self.output_mode == 'PIL':
            img = Image.open(fname)
        else:
            raise NotImplementedError
        return dict(name=Path(fname).stem, img=img)

    def max_bsz_sfd(self, gpu_idx):
        return 2 ** int(math.log2(int(torch.cuda.mem_get_info(gpu_idx)[1] / 1024 ** 2 - 361) / (
                0.000696875 * read_image(self.img_lists[0]).numel())))

    def max_bsz_retinaface(self, gpu_idx):
        return 2 ** int(math.log2(int(torch.cuda.mem_get_info(gpu_idx)[1] / 1024 ** 2 - 620) / (
                0.00015 * read_image(self.img_lists[0]).numel())))

    def pop(self) -> [AnyStr, torch.Tensor]:
        fname = self.img_lists.pop()
        return Path(fname).stem, read_image(fname)


class GenerateDataset(Dataset):

    def __init__(self, folder, mel_spec,
                 window_size=5,
                 fps=25, mel_step_size=16,
                 ext='jpg', face_size=(256, 256)) -> None:
        super().__init__()
        self.face_lists = cycle(sorted(glob(f"{folder}/crop_face/*.{ext}")))
        self.frame_lists = cycle(sorted(glob(f"{folder}/frames/*.{ext}")))
        self.coords = dict()
        with open(os.path.join(folder, 'meta.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                name, bbox = line.rstrip('\n').split(' ')
                self.coords[name] = list(map(lambda v: eval(v), bbox.split(',')))
        # self.mel_spec = mel_spec

        self.window_size = window_size
        self.fps = fps
        self.mel_step_size = mel_step_size
        self.face_size = face_size

        self.mel_chunks = []
        mel_idx_multiplier = 80. / fps
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > mel_spec.shape[0]:
                self.mel_chunks.append(mel_spec[mel_spec.shape[0] - mel_step_size:, :])
                break
            self.mel_chunks.append(mel_spec[start_idx: start_idx + mel_step_size, :])
            i += 1
        print("Length of mel chunks: {}".format(len(self.mel_chunks)))

    def __len__(self):
        return len(self.mel_chunks)

    def __getitem__(self, index: int) -> [torch.Tensor, torch.Tensor, torch.Tensor, AnyStr]:
        index = index

        # fnames = [next(self.face_lists) for _ in range(self.window_size)]
        # ori_frames = [next(self.frame_lists) for _ in range(self.window_size)]

        # meta_name = ','.join([(Path(name).stem) for name in fnames])

        # # window = []

        # # for fname in fnames:
        # #     img =  resize(read_image(fname), self.face_size,
        # #                  interpolation=InterpolationMode.BILINEAR,
        # #                  antialias=True)
        # #     window.append(img)

        # ref = torch.stack(window, dim=1) / 255
        # mask_window = ref_window.clone()
        # mask_window[:, :, mask_window.size(2) // 2:] = 0
        # x = torch.cat([mask_window, ref_window], dim=0)

        # indiv_mels = self._segmented_mels(self.mel_spec.copy(), index)
        # indiv_mels = torch.tensor(indiv_mels, dtype=torch.float).unsqueeze(1)

        # ori_window = torch.stack([torch.tensor(cv2.imread(fname)) for fname in ori_frames], dim=1)

        # return x, indiv_mels, ori_window, meta_name
        fname = next(self.face_lists)
        ori_frame = next(self.frame_lists)

        meta_name = Path(fname).stem

        img = resize(read_image(fname), self.face_size,
                     interpolation=InterpolationMode.BILINEAR,
                     antialias=True)

        ref = img / 255
        mask = ref.clone()
        y1, y2, x1, x2 = [16, -16, 48, -48]
        mask[:, mask.size(1) // 2 + y1:y2, x1:x2] = 0
        # mask[:, mask.size(1) // 2:, :] = 0

        x = torch.cat([mask, ref], dim=0)

        mel = torch.tensor(self.mel_chunks[index].T, dtype=torch.float).unsqueeze(0)
        # mel = torch.tensor(self._crop_audio_window(self.mel_spec, index).T, dtype=torch.float).unsqueeze(0)

        ori = torch.tensor(cv2.imread(ori_frame))

        return x, mel, ori, meta_name


def get_largest_face(det_faces, h, w):
    def get_location(val, length):
        if val < 0:
            return 0
        elif val > length:
            return length
        else:
            return val

    face_areas = []
    for det_face in det_faces:
        left = get_location(det_face[0], w)
        right = get_location(det_face[2], w)
        top = get_location(det_face[1], h)
        bottom = get_location(det_face[3], h)
        face_area = (right - left) * (bottom - top)
        face_areas.append(face_area)
    largest_idx = face_areas.index(max(face_areas))
    return det_faces[largest_idx], largest_idx


def get_center_face(det_faces, h=0, w=0, center=None):
    if center is not None:
        center = np.array(center)
    else:
        center = np.array([w / 2, h / 2])
    center_dist = []
    for det_face in det_faces:
        face_center = np.array([(det_face[0] + det_face[2]) / 2, (det_face[1] + det_face[3]) / 2])
        dist = np.linalg.norm(face_center - center)
        center_dist.append(dist)
    center_idx = center_dist.index(min(center_dist))
    return det_faces[center_idx], center_idx


class EMA:

    def __init__(self, decay=0.9):
        self._avg_value = None
        self.n_average = 1
        self.decay = decay

    def update(self, new_value):
        if self._avg_value is None:
            self._avg_value = copy.deepcopy(new_value)
            self.n_average = 1
        else:
            decay = self.decay
            # decay = 2/(self.n_average+1)
            # self._avg_value = self._avg_value + (new_value - self._avg_value) / (self.n_average + 1)
            self._avg_value = decay * self._avg_value + new_value * (1 - decay)

            self.n_average += 1

    @property
    def avg_value(self):
        return self._avg_value

    @property
    def recent(self):
        return 1 / (1 - self.decay)
