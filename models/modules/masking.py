import random
from pathlib import Path

from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize
from torchvision.ops import masks_to_boxes

from arch.segmentation import BiSeNet


class Masking(nn.Module):

    def __init__(self, size=256, pad=0.02, half_precision=True, norm=True):
        """
        Masking toolkit for half-face masking
        Args:
            size: mask image size
            half_precision: whether using half-precision inference
        """
        super().__init__()
        self.detector = BiSeNet(n_classes=19, norm=norm)
        assert 0 <= pad < 0.5, f"mask pad should between [0, 0.5], but got {pad}"
        self.pad = pad  # pad mask for better boundary effect.
        self.half_precision = half_precision
        ckpt_path = Path(__file__).parent / f"weights/face_parsing_{size}.pth"
        self.detector.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        self.detector.eval()
        self.inverse_mask = None
        self.use_amp = half_precision

    def mask(self, x, mask_face=True, bottom=True, lip_only=False):
        """
        Args:
            x: input image
            mask_face: whether mask face region (True mask out face, False preserve face only)
            bottom: bottom face only mask
            lip_only: if return reshaped mask_lip
        Returns:
            masks
        """

        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else 'cpu',
                            dtype=torch.float16 if torch.cuda.is_available() else torch.bfloat16,
                            enabled=self.use_amp):
            mask = self.detector(x).argmax(1)

        flag = (mask == 1) | (mask == 2) | (mask == 3) | (mask == 4) | (mask == 5) | (mask == 6) | (mask == 10) | (
                mask == 11) | (mask == 12) | (mask == 13)

        face_masks = torch.where(flag, torch.zeros_like(mask), 1)
        if bottom:
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
            if lip_only:
                assert face_masks.size(0) % 5 == 0, f"get shape {face_masks.size()}"
                for idx, face_mask in enumerate(face_masks):
                    if idx % 5 == 0:
                        try:
                            x1, y1, x2, y2 = masks_to_boxes(1 - face_mask.unsqueeze(0)).int().tolist()[0]
                            # x[idx:idx + 5] = F.interpolate((x[idx:idx + 5] * (1 - face_masks[idx:idx + 5, None]))[..., y1:y2, x1:x2], [256, 256])
                            x[idx:idx + 5, :, :128, :] = F.interpolate((x[idx:idx + 5])[..., y1:y2, x1:x2], [128, 256])
                        except RuntimeError as e:
                            # if error use bottom half
                            x[idx:idx + 5] = F.interpolate((x[idx:idx + 5])[..., 128:, :], [256, 256])

                # return F.interpolate(x, [128, 256])
                return x[:, :, :128, ]
        mask = face_masks.unsqueeze(1)
        if mask_face:
            self.inverse_mask = (1 - mask)
            return x * mask
        else:
            self.inverse_mask = mask
            return x * (1 - mask)

    def forward(self, x: torch.Tensor, mask_face=True, bottom=True, lip_only=False) -> torch.Tensor:
        """

        Args:
            x: input image
            mask_face: whether mask face region (True mask out face, False preserve face only)
            bottom: bottom face only mask
            lip_only: if return reshaped mask_lip
        Returns:
            masked image
        """
        bsz = x.size(0)
        x = x.clone()
        if x.dim() == 5:
            assert x.size(1) == 6
            # face [bsz, 6, t, h, w]
            face, ref = x.split(3, dim=1)
            face = rearrange(face, 'b c t h w -> (b t) c h w')
            face = self.mask(face, mask_face, bottom, lip_only)
            face = rearrange(face, '(b t) c h w -> b c t h w ', b=bsz)
            return torch.cat([face, ref], dim=1)
        elif x.dim() == 4:
            if x.size(1) == 6:
                face, ref = x.split(3, dim=1)
                face = self.mask(face, mask_face, bottom, lip_only)
                return torch.cat([face, ref], dim=1)
            elif x.size(1) == 3:
                x = self.mask(x, mask_face, bottom, lip_only)
                return x
            elif x.size(1) == 15:
                face = rearrange(x, 'b (t c) h w -> (b t) c h w', c=3)
                face = self.mask(face, mask_face, bottom, lip_only)
                return rearrange(face, '(b t) c h w -> b (t c) h w', b=bsz)
            else:
                raise NotImplementedError(f'{x.shape}')
        else:
            raise NotImplementedError(f'{x.dim()}')
