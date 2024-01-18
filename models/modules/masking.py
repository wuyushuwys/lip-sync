import random
from pathlib import Path

from einops import rearrange

import torch
import torch.nn as nn
from torchvision.ops import masks_to_boxes

from arch.segmentation import BiSeNet


class Masking(nn.Module):

    def __init__(self, size=256, pad=0.02, half_precision=True):
        """
        Masking toolkit for half-face masking
        Args:
            size: mask image size
            half_precision: whether using half-precision inference
        """
        super().__init__()
        self.detector = BiSeNet(n_classes=19)
        assert 0 <= pad < 0.5, f"mask pad should between [0, 0.5], but got {pad}"
        self.pad = pad  # pad mask for better boundary effect.
        self.half_precision = half_precision
        ckpt_path = Path(__file__).parent / f"weights/face_parsing_{size}.pth"
        self.detector.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        self.detector.eval()

        if half_precision:
            self.detector.half()

    @torch.no_grad()
    def mask(self, x):

        mask = self.detector(x).argmax(1)

        flag = (mask == 1) | (mask == 2) | (mask == 3) | (mask == 4) | (mask == 5) | (mask == 6) | (mask == 10) | (
                mask == 11) | (mask == 12) | (mask == 13)

        face_masks = torch.where(flag, torch.zeros_like(mask), 1)
        nose_masks = torch.where(mask == 10, torch.ones_like(mask), 0)

        try:
            nose_bbox = masks_to_boxes(nose_masks).int().tolist()
        except RuntimeError as e:
            nose_bbox = []
            for nose_mask in nose_masks:
                try:
                    nose_bbox.append(masks_to_boxes(nose_mask.unsqueeze(0)).int().tolist()[0])
                except RuntimeError as e:
                    nose_bbox.append(None)
            # for nose_mask, face_mask in zip(nose_masks, face_masks):
            #     try:
            #         nose_bbox.append(masks_to_boxes(nose_mask.unsqueeze(0)).int().tolist()[0])
            #     except RuntimeError as e:
            #         try:
            #             x1, y1, x2, y2 = masks_to_boxes(face_mask.unsqueeze(0)).int().tolist()[0]
            #             nose_bbox.append([x1, y1, x2, (y2 * 3) // 5])
            #         except RuntimeError as e:
            #             x1, y1, x2, y2 = 0, 0, face_masks.size(2), face_masks.size(1) // 2
            #             nose_bbox.append([x1, y1, x2, y2])

        # nose_bound = masks_to_boxes(nose_mask)[:-1].mean().int().item()

        # face_mask[:, :nose_bound, ...] = 1
        # h = random.randint(1, int(face_masks.size(1) * self.pad))
        # w = random.randint(1, int(face_masks.size(2) * self.pad))
        h = int(face_masks.size(1) * self.pad)
        w = int(face_masks.size(2) * self.pad)
        for idx, bbox in enumerate(nose_bbox):
            if bbox is None:
                face_masks[idx] = 1  # no mask if failed to detect nose (normally caused by bad face detection)
            else:
                x1, y1, x2, y2 = bbox
                face_masks[idx, :y2, ...] = 1
                face_masks[idx, -h:, ...] = 1
                face_masks[idx, ..., :w] = 1
                face_masks[idx, ..., -w:] = 1

        return face_masks.unsqueeze(1)

    def forward(self, x: torch.Tensor):
        bsz = x.size(0)
        if x.dim() == 5:
            assert x.size(1) == 6
            # face [bsz, 6, t, h, w]
            face, ref = x.split(3, dim=1)
            face = rearrange(face, 'b c t h w -> (b t) c h w')
            face = face * self.mask(face.half() if self.half_precision else face)
            face = rearrange(face, '(b t) c h w -> b c t h w ', b=bsz)
            return torch.cat([face, ref], dim=1)
        else:
            if x.size(1) == 6:
                face, ref = x.split(3, dim=1)
                face = face * self.mask(face.half() if self.half_precision else face)
                return torch.cat([face, ref], dim=1)
            elif x.size(1) == 3:
                x = x * self.mask(x.half() if self.half_precision else x)
                return x