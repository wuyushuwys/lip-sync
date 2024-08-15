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
from einops import rearrange
EXT = 'jpg'
TMP_FOLDER = 'tmp_vq'
SAMPLE_RATE = 16000
FPS = 25
SIZE = 256

face_template = np.array([[192.98138, 239.94708],
                          [318.90277, 240.1936],
                          [256.63416, 314.01935],
                          [201.26117, 371.41043],
                          [313.08905, 371.15118]]) / 512 * SIZE

meta_line = lambda name, bbox, lm, inv_affine: f"{name} {','.join(map(str, bbox))} " \
                                               f"{','.join(lm.flatten().astype(str).tolist())} " \
                                               f"{','.join(inv_affine.flatten().astype(str).tolist())}\n"

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
    os.makedirs(os.path.join(TMP_FOLDER, 'align_face'), exist_ok=True)
    dataset = ImageFolder(os.path.join(TMP_FOLDER, 'frames'), output_mode='cv2')
    print(f"Total frame extracted {len(dataset)}")
    face_detector = init_detection_model(model_name='retinaface_resnet50', half=True,
                                         device='cuda' if torch.cuda.is_available() else 'cpu')
    bsz = dataset.max_bsz_retinaface(0)
    dataloader = DataLoader(dataset, batch_size=bsz, num_workers=8, prefetch_factor=10)
    with open(os.path.join(TMP_FOLDER, 'meta.txt'), 'w') as f:
        for data in tqdm(dataloader, total=len(dataloader), desc='face extraction', dynamic_ncols=True):
            names, imgs = data['name'], data['img']
            batched_bboxes, batched_lms = face_detector.batched_detect_faces(imgs,
                                                                             conf_threshold=0.97,
                                                                             nms_threshold=0.3,
                                                                             use_origin_size=True)
            b, h, w, c = imgs.size()
            batched_det_faces = []
            batched_det_lm = []
            for bboxes, lms in zip(batched_bboxes, batched_lms):
                det_faces = []
                det_lms = []
                if len(bboxes) == 0:
                    batched_det_faces.append(None)
                    batched_det_faces.append(None)
                else:
                    for bbox, lm in zip(bboxes, lms):
                        det_faces.append(bbox[0:5])
                        det_lms.append(lm)
                    det_faces, idx = get_largest_face(det_faces, h=h, w=w)
                    batched_det_faces.append(det_faces.astype(int).tolist()[:4])
                    batched_det_lm.append(np.array(np.split(lm, 5, axis=0)))
            for name, bbox, landmark, img in zip(names, batched_det_faces, batched_det_lm, imgs):
                output_path_crop = os.path.join(TMP_FOLDER, 'crop_face', f'{name}.{EXT}')
                output_path_align = os.path.join(TMP_FOLDER, 'align_face', f'{name}.{EXT}')

                if bbox:
                    x1, y1, x2, y2 = bbox
                    cropped_face = img.numpy()[y1:y2, x1:x2, :]
                    affine_matrix = cv2.estimateAffinePartial2D(landmark, face_template, method=cv2.LMEDS)[0]
                    aligned_face = cv2.warpAffine(img.numpy(), affine_matrix, [SIZE, SIZE],
                                                  borderMode=cv2.BORDER_CONSTANT,
                                                  borderValue=(135, 133, 132))
                    inv_affine = cv2.invertAffineTransform(affine_matrix)
                    cv2.imwrite(output_path_crop, cropped_face)
                    cv2.imwrite(output_path_align, aligned_face)
                    f.write(meta_line(name, bbox, landmark, inv_affine))
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
    dataset = GenerateDataset(TMP_FOLDER, mel, landmark=True, dynamic_mask=args.dynamic_mask, chunk=True, chunk_size=5)
    coords = dataset.coords
    landmarks = dataset.landmarks
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FaceCoderTemporalNet(
        in_channel=3,
        codebook_size=512,
        emb_dim=256,
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
               s='{}x{}'.format(512, 256),
               r=FPS,
               thread_queue_size=1024)
        .output(args.output,
                pix_fmt="yuv420p",
                vcodec="libx264",
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
        batch_size = x.shape[0]
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        with torch.no_grad():
            g, _ = model(x.half())
            g = g.clamp(-1, 1) / 2 + 0.5
            g = rearrange(g, '(b t) c h w -> b t c h w', b=batch_size)
            x = x.clamp(-1, 1) / 2 + 0.5
            x = rearrange(x, '(b t) c h w -> b t c h w', b=batch_size)

        for batch_id, (faces, gts) in enumerate(zip(g.unbind(0), x.unbind(0))):
            for face, gt in zip(faces.unbind(), gts.unbind()):
                output = torch.cat([face, gt], dim=-1)
                output = (output * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                process.stdin.write(output.astype(np.uint8).tobytes())

    process.stdin.close()
    process.wait()

    if args.clean:
        shutil.rmtree(TMP_FOLDER)
