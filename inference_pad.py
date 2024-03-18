import warnings

warnings.filterwarnings("ignore")
import argparse
import cv2
import os
import shutil
import numpy as np

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import ffmpeg
from facexlib.detection import init_detection_model

from inference_utils import ImageFolder, get_largest_face, GenerateDataset, EMA
from arch.wav2lip_arch import Wav2Lip
from models.modules.masking import Masking
from utils import audio

EXT = 'jpg'
TMP_FOLDER = 'tmp_pad'
SAMPLE_RATE = 16000
FPS = 25

meta_line = lambda name, x1, y1, x2, y2: f"{name} {x1},{y1},{x2},{y2}\n"

parser = argparse.ArgumentParser(description='lip-sync')
parser.add_argument('--input', type=str, required=True, help='input image or video input')
parser.add_argument('--audio', type=str, required=True, help='input audio')
parser.add_argument('--ckpt', type=str, required=True, help='model ckpt path')
parser.add_argument('--output', type=str, default='output.mp4', help='output video path')
parser.add_argument('--verbose', action='store_true', help='whether save results during generation for debug')
parser.add_argument('--clean', action='store_true', help='whether clean intermedia results afterwards')
parser.add_argument('--smooth', action='store_true', help='smooth face detection')
parser.add_argument('--dynamic-mask', '-mask', action='store_true', help='dynamic masking face')

args = parser.parse_args()

TMP_FOLDER = os.path.join(TMP_FOLDER, '_'.join(os.path.splitext(args.input)[0].split('/')))
if args.smooth:
    TMP_FOLDER += '_smooth'


def extract_frames(file_path):
    os.makedirs(os.path.join(TMP_FOLDER, 'frames'), exist_ok=True)
    streams = ffmpeg.input(file_path)
    streams.video.output(os.path.join(TMP_FOLDER, 'frames', f'%06d.{EXT}'),
                         **{'qscale:v': 0, 'r': FPS}).run(overwrite_output=True, quiet=True)


@torch.no_grad()
def face_crop():
    os.makedirs(os.path.join(TMP_FOLDER, 'crop_face'), exist_ok=True)
    dataset = ImageFolder(os.path.join(TMP_FOLDER, 'frames'), output_mode='cv2')
    print(f"Total frame extracted {len(dataset)}")
    face_detector = init_detection_model(model_name='retinaface_resnet50', half=True,
                                         device='cuda' if torch.cuda.is_available() else 'cpu')
    bsz = dataset.max_bsz_retinaface(0)
    dataloader = DataLoader(dataset, batch_size=bsz, num_workers=8, prefetch_factor=10)
    if args.smooth:
        bbox_ema = EMA()
    with open(os.path.join(TMP_FOLDER, 'meta.txt'), 'w') as f:
        for data in tqdm(dataloader, total=len(dataloader), desc='face extraction', dynamic_ncols=True):
            names, imgs = data['name'], data['img']
            batched_bboxes, _ = face_detector.batched_detect_faces(imgs,
                                                                   conf_threshold=0.97,
                                                                   nms_threshold=0.3,
                                                                   use_origin_size=True)
            b, h, w, c = imgs.size()
            batched_det_faces = []

            for bboxes in batched_bboxes:
                det_faces = []
                if len(bboxes) == 0:
                    batched_det_faces.append(None)
                else:
                    for bbox in bboxes:
                        det_faces.append(bbox[0:5])
                    det_faces, _ = get_largest_face(det_faces, h=h, w=w)
                    batched_det_faces.append(det_faces.astype(int).tolist()[:4])

            for name, bbox, img in zip(names, batched_det_faces, imgs):
                output_path = os.path.join(TMP_FOLDER, 'crop_face', f'{name}.{EXT}')
                if bbox:
                    if args.smooth:
                        bbox_ema.update(np.array(bbox))
                        x1, y1, x2, y2 = bbox_ema.avg_value.astype(int).tolist()
                    else:
                        x1, y1, x2, y2 = bbox
                    hh = y2 - y1
                    hw = x2 - x1
                    h_pad = 0.1
                    w_pad = 0.25
                    x1 = int(x1 - w_pad * hw) if x1 - w_pad * hw >=0 else 0
                    y1 = int(y1 - h_pad * hh) if y1 - h_pad * hh >= 0 else 0
                    x2 = int(x2 + w_pad * hw)
                    y2 = int(y2 + h_pad * hw)
                    bbox = x1, y1, x2, y2
                    cropped_face = img.numpy()[y1:y2, x1:x2, :]
                    cv2.imwrite(output_path, cropped_face)
                    f.write(meta_line(name, *bbox))
    del face_detector
    return h, w


if __name__ == '__main__':
    if not os.path.exists(TMP_FOLDER):
        extract_frames(args.input)
        h, w = face_crop()
    else:
        frame_dir = os.path.join(TMP_FOLDER, 'frames')
        img_file = [fname for fname in os.listdir(frame_dir) if fname.endswith(EXT)][0]
        h, w, _ = cv2.imread(os.path.join(frame_dir, img_file)).shape
    print(f"Video Resolution {h}x{w}")
    # h, w = 720, 1280
    # h, w = 1080, 1920
    mel = audio.melspectrogram(audio.load_wav(path=args.audio, sr=SAMPLE_RATE)).T
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')
    dataset = GenerateDataset(TMP_FOLDER, mel, dynamic_mask=args.dynamic_mask)
    coords = dataset.coords
    # win_size = dataset.window_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.dynamic_mask:
        mask = Masking(half_precision=True, norm=False).to(device)
    model = Wav2Lip()
    model.load_state_dict(torch.load(args.ckpt))
    model.to(device)
    model.half()
    model.eval()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, prefetch_factor=8, num_workers=8)

    os.makedirs(os.path.join(TMP_FOLDER, 'sync_face'), exist_ok=True)
    os.makedirs(os.path.join(TMP_FOLDER, 'sync_frames'), exist_ok=True)
    os.makedirs(os.path.join(TMP_FOLDER, 'diff'), exist_ok=True)
    os.makedirs(os.path.join(TMP_FOLDER, 'compare'), exist_ok=True)
    os.makedirs(os.path.join(TMP_FOLDER, 'mask'), exist_ok=True)
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo',
               pix_fmt='rgb24',
               s='{}x{}'.format(w, h),
               r=FPS,
               thread_queue_size=1024)
        .output(ffmpeg.input(args.audio, channel_layout="stereo"),
                args.output,
                pix_fmt="yuv420p",
                vcodec="libx264",
                acodec='aac',
                r=FPS,
                crf=18,
                )
        .overwrite_output()
        .run_async(pipe_stdin=True, quiet=not args.verbose)
    )

    if args.verbose:
        pbar = enumerate(dataloader)
    else:
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='lip-sync', dynamic_ncols=True)
    for i, (x, indiv_mels, ori_window, meta) in pbar:
        x = x.to(device, non_blocking=True)
        x = x * 2 - 1
        indiv_mels = indiv_mels.to(device, non_blocking=True)
        bsz = x.size(0)
        if args.dynamic_mask:
            with torch.no_grad():
                x = mask(x)
            masked_flag = mask.inverse_mask.cpu()
        with torch.no_grad():
            g = model(indiv_mels.half(), x.half())
            g = g.clamp(-1, 1) / 2 - 0.5
        for batch_id, (face, frame, name) in enumerate(zip(g.unbind(0), ori_window.unbind(0), meta)):
            frame_idx = i * bsz + batch_id
            x1, y1, x2, y2 = coords[name]
            frame = frame.flip(-1).numpy()
            face = (face * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            if args.dynamic_mask:
                ori_face = cv2.resize(frame[y1:y2, x1:x2], dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
                face_mask = masked_flag[batch_id, ...].squeeze().cpu().numpy().astype(np.float32)
                inv_mask_erosion = cv2.erode(face_mask, np.ones((2, 2), np.uint8))
                # pasted_face = inv_mask_erosion[:, :, None] * face
                pasted_face = face
                total_face_area = np.sum(inv_mask_erosion)  # // 3
                w_edge = int(total_face_area ** 0.5) // 20
                erosion_radius = w_edge * 2
                inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
                blur_size = w_edge * 2
                inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)
                inv_soft_mask = inv_soft_mask[:, :, None]
                face = (inv_soft_mask * pasted_face + (1 - inv_soft_mask) * ori_face).astype(np.uint8)
                # face = (face * face_mask + ori_face * (1 - face_mask)).astype(np.uint8)
                # face = (face * face_mask).astype(np.uint8)
            if args.verbose:
                g = face.copy()
                ref = cv2.resize(frame[y1:y2, x1:x2], (256, 256))
                concat = np.flip(np.concatenate([ref, g], axis=1), axis=-1)
                diff = np.abs(g.astype(float) - ref.astype(float))
                diff /= diff.max()  # diff.mean(axis=-1, keepdims=True) * 255
                cv2.imwrite(f'{TMP_FOLDER}/diff/{frame_idx:06d}.jpg', np.abs(diff).astype(np.uint8))
                cv2.imwrite(f'{TMP_FOLDER}/compare/{frame_idx:06d}.jpg', concat.astype(np.uint8))
                cv2.imwrite(f'{TMP_FOLDER}/mask/{frame_idx:06d}.jpg', (face_mask * 255).astype(np.uint8))
            # w_offset = int(48 / 256 * (x2 - x1))
            # h_offset = int(64 / 256 * (y2 - y1))
            resize_face = cv2.resize(face, dsize=(x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)
            # frame[y1:y2 - h_offset, x1 + w_offset:x2 - w_offset] = resize_face[:-h_offset, w_offset:-w_offset]
            frame[y1:y2, x1:x2] = resize_face
            if args.verbose:
                cv2.imwrite(os.path.join(TMP_FOLDER, 'sync_face', f"{frame_idx:06d}.png"), np.flip(resize_face, -1))
                cv2.imwrite(os.path.join(TMP_FOLDER, 'sync_frames', f"{frame_idx:06d}.jpg"), np.flip(frame, -1))
                cv2.imwrite(f'{TMP_FOLDER}/mask/{frame_idx:06d}.jpg', (face_mask * 255).astype(np.uint8))
            process.stdin.write(frame.astype(np.uint8).tobytes())

    process.stdin.close()
    process.wait()

    if args.clean:
        shutil.rmtree(TMP_FOLDER)
