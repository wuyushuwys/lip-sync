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
# from torchvision.transforms.functional import resize
from torchvision.utils import save_image

import ffmpeg
from facexlib.detection import init_detection_model

from inference_utils import ImageFolder, get_largest_face, GenerateDataset, EMA
from models.wav2lip import Wav2Lip
from utils import audio

EXT = 'jpg'
TMP_FOLDER = 'tmp'
SAMPLE_RATE = 16000
FPS = 25

meta_line = lambda name, x1, y1, x2, y2: f"{name} {x1},{y1},{x2},{y2}\n"

parser = argparse.ArgumentParser(description='lip-sync')
parser.add_argument('--input', type=str, required=True, help='input image or video input')
parser.add_argument('--audio', type=str, required=True, help='input audio')
parser.add_argument('--ckpt', type=str, required=True, help='model ckpt path')
parser.add_argument('--output', type=str, default='output.mp4', help='output video path')
parser.add_argument('--verbose', action='store_true', help='whether save results during generation for debug')

args = parser.parse_args()


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
    dataloader = DataLoader(dataset, batch_size=bsz)
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
                    bbox_ema.update(np.array(bbox))
                    x1, y1, x2, y2 = bbox_ema.avg_value.astype(int).tolist()
                    cropped_face = img.numpy()[y1:y2, x1:x2, :]
                    cv2.imwrite(output_path, cropped_face)
                    f.write(meta_line(name, *bbox))
    del face_detector
    return h, w


if __name__ == '__main__':
    if os.path.exists(TMP_FOLDER): shutil.rmtree(TMP_FOLDER)
    extract_frames(args.input)
    h, w = face_crop()
    # h, w = 720, 1280
    mel = audio.melspectrogram(audio.load_wav(path=args.audio, sr=SAMPLE_RATE)).T

    dataset = GenerateDataset(TMP_FOLDER, mel)
    coords = dataset.coords
    # win_size = dataset.window_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Wav2Lip()
    model.load_state_dict(torch.load(args.ckpt))
    model.to(device)
    model.half()
    model.eval()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    os.makedirs(os.path.join(TMP_FOLDER, 'sync_face'), exist_ok=True)
    os.makedirs(os.path.join(TMP_FOLDER, 'sync_frames'), exist_ok=True)
    os.makedirs(os.path.join(TMP_FOLDER, 'diff'), exist_ok=True)
    os.makedirs(os.path.join(TMP_FOLDER, 'compare'), exist_ok=True)
    tmp_vname = f'{TMP_FOLDER}/result.mp4'
    # tmp_video = cv2.VideoWriter(tmp_vname,  cv2.VideoWriter_fourcc(*'DIVX'), FPS, (w, h))
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
        .run_async(pipe_stdin=True, quiet=False)
    )
    # pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='lip-sync', dynamic_ncols=True)
    for i, data in enumerate(dataloader):
        x, indiv_mels, ori_window, meta = data
        x = x.to(device, non_blocking=True)
        indiv_mels = indiv_mels.to(device, non_blocking=True)
        bsz = x.size(0)
        with torch.no_grad():
            g = model(indiv_mels.half(), x.half()).clamp(0, 1)
        # for batch_id, (_g, _ori, names) in enumerate(zip(g.unbind(0), ori_window.unbind(0), meta)):
        for batch_id, (face, frame, name) in enumerate(zip(g.unbind(0), ori_window.unbind(0), meta)):
            frame_idx = i * bsz + batch_id
            x1, y1, x2, y2 = coords[name]
            frame = frame.flip(-1).numpy()
            # save_image(face, os.path.join(TMP_FOLDER, 'sync_face', f"{frame_idx:06d}.png"))
            face = (face * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            if args.verbose:
                g = face.copy()
                ref = cv2.resize(frame[y1:y2, x1:x2], (256, 256))
                concat = np.flip(np.concatenate([ref, g], axis=1), axis=-1)
                diff = np.abs(g.astype(float) - ref.astype(float))
                # diff /= diff.max() diff.mean(axis=-1, keepdims=True) * 255
                cv2.imwrite(f'{TMP_FOLDER}/diff/{frame_idx:06d}.jpg', np.abs(diff).astype(np.uint8))
                cv2.imwrite(f'{TMP_FOLDER}/compare/{frame_idx:06d}.jpg', concat.astype(np.uint8))
            # w_offset = int(48 / 256 * (x2-x1))
            # h_offset = int(16 / 256 * (y2-y1))
            # resize_face = cv2.resize(face, dsize=(x2-x1, y2-y1))[:-h_offset, w_offset:-w_offset]
            resize_face = cv2.resize(face, dsize=(x2 - x1, y2 - y1))
            if args.verbose:
                cv2.imwrite(os.path.join(TMP_FOLDER, 'sync_face', f"{frame_idx:06d}.png"), np.flip(resize_face, -1))
            # frame[y1:y2-h_offset,x1+w_offset:x2-w_offset] = resize_face
            frame[y1:y2, x1:x2] = resize_face
            if args.verbose:
                cv2.imwrite(os.path.join(TMP_FOLDER, 'sync_frames', f"{frame_idx:06d}.jpg"), np.flip(frame, -1))
            # tmp_video.write(frame.astype(uint8))
            process.stdin.write(frame.astype(np.uint8).tobytes())

    process.stdin.close()
    process.wait()
    # ffmpeg.output(ffmpeg.input(tmp_vname),
    #               ffmpeg.input(args.audio),
    #               args.output).run(overwrite_output=True, quiet=False)
    if args.verbose:
        shutil.rmtree(TMP_FOLDER)