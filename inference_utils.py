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
                 ext='jpg', face_size=(256, 256),
                 dynamic_mask=False, landmark=False, mage=False, chunk=False, chunk_size=None) -> None:
        super().__init__()
        face_folder = 'crop_face' if not landmark else 'align_face'
        self.face_lists = sorted(glob(os.path.join(folder, face_folder, f"*.{ext}")))
        self.frame_lists = sorted(glob(f"{folder}/frames/*.{ext}"))
        self.num_video_frames = len(self.frame_lists)
        self.dynamic_mask = dynamic_mask
        self.landmark = landmark
        self.mage = mage
        self.chunk = chunk
        self.chunk_size = chunk_size
        self.coords = dict()
        self.landmarks = dict()
        self.inv_affine_matrices = dict()
        with open(os.path.join(folder, 'meta.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                if landmark:
                    name, bbox, lm, inv_affine = line.rstrip('\n').split(' ')
                    self.landmarks[name] = np.split(np.fromstring(lm, dtype=np.float32, sep=','), 5, axis=0)
                    self.inv_affine_matrices[name] = np.array(
                        np.split(np.fromstring(inv_affine, dtype=np.float32, sep=','), 2, axis=0))
                else:
                    name, bbox = line.rstrip('\n').split(' ')
                self.coords[name] = list(map(lambda v: eval(v), bbox.split(',')))

                # print(np.fromstring(lm, dtype=np.float32, sep=',').reshape(5, 2))
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
        # print("Length of mel chunks: {}".format(len(self.mel_chunks)))

    def __len__(self):
        if self.chunk:
            return len(self.mel_chunks) // self.chunk_size
        else:
            return len(self.mel_chunks)

    def __getitem__(self, index: int) -> [torch.Tensor, torch.Tensor, torch.Tensor, AnyStr]:
        if not self.chunk:
            index = index
            fname = self.face_lists[index % self.num_video_frames]
            ori_frame = self.frame_lists[index % self.num_video_frames]

            meta_name = Path(fname).stem

            img = resize(read_image(fname), self.face_size,
                         interpolation=InterpolationMode.BILINEAR,
                         antialias=True)

            ref = img / 255
            mask = ref.clone()
            if not self.dynamic_mask:
                y1, y2, x1, x2 = [16, -16, 48, -48]
                mask[:, mask.size(1) // 2 + y1:y2, x1:x2] = 0
            # mask[:, mask.size(1) // 2:, :] = 0
            if not self.mage:
                x = torch.cat([mask, ref], dim=0)
            else:
                x = ref * 2 - 1

            mel = torch.tensor(self.mel_chunks[index].T, dtype=torch.float).unsqueeze(0)
            # mel = torch.tensor(self._crop_audio_window(self.mel_spec, index).T, dtype=torch.float).unsqueeze(0)

            ori = torch.tensor(cv2.imread(ori_frame))

            return x, mel, ori, meta_name
        else:
            assert self.chunk_size > 0, self.chunk_size
            frames = []
            cid = index * self.chunk_size
            for i in range(self.chunk_size):
                id = cid + i
                fname = self.face_lists[id % self.num_video_frames]
                img = resize(read_image(fname), self.face_size,
                             interpolation=InterpolationMode.BILINEAR,
                             antialias=True)
                img = img / 255
                frames.append(img)
            frames = torch.stack(frames)
            frames = frames * 2 - 1
            return frames

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
