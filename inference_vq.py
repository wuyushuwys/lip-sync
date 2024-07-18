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
from arch.fema_temporal_vqgan_arch import FaceCoderTemporalNet
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
                    x1 = int(x1 - w_pad * hw) if x1 - w_pad * hw >= 0 else 0
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
    mel = audio.melspectrogram(audio.load_wav(path=args.audio, sr=SAMPLE_RATE)).T
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')
    dataset = GenerateDataset(TMP_FOLDER, mel, dynamic_mask=args.dynamic_mask, chunk=True, chunk_size=5)
    coords = dataset.coords
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FaceCoderTemporalNet(
        in_channel=3,
        codebook_size=1024,
        codebook_scale=512,
    )
    model.load_state_dict(torch.load(args.ckpt))
    model.to(device)
    model.half()
    model.eval()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, prefetch_factor=8, num_workers=8)

    process = (
        ffmpeg
        .input('pipe:', format='rawvideo',
               pix_fmt='rgb24',
               s='{}x{}'.format(256, 256),
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
    for i, x in pbar:
        x = x.to(device, non_blocking=True)
        with torch.no_grad():
            g = model(x.half())
            g = g.clamp(-1, 1) / 2 + 0.5
        for batch_id, faces in enumerate(g.unbind(0)):
            for face in faces.unbind():
                face = (face * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                process.stdin.write(face.astype(np.uint8).tobytes())

    process.stdin.close()
    process.wait()

    if args.clean:
        shutil.rmtree(TMP_FOLDER)
