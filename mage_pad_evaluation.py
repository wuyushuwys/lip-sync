import warnings

warnings.filterwarnings("ignore")
import argparse
import cv2
import os
import shutil
import numpy as np
import datetime
import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import ffmpeg
from facexlib.detection import init_detection_model

from inference_utils import ImageFolder, get_largest_face, GenerateDataset
from arch.conditioned_mage_arch import lip_mage_vit_base
from arch.modules.masking import Masking
from utils import audio

EXT = 'jpg'
TMP_FOLDER = 'tmp_mage'
SAMPLE_RATE = 16000
FPS = 25
SIZE = 256
batch_size = 16

face_template = np.array([[192.98138, 239.94708],
                          [318.90277, 240.1936],
                          [256.63416, 314.01935],
                          [201.26117, 371.41043],
                          [313.08905, 371.15118]]) / 512 * SIZE

meta_line = lambda name, x1, y1, x2, y2: f"{name} {x1},{y1},{x2},{y2}\n"

parser = argparse.ArgumentParser(description='lip-sync')
parser.add_argument('--input_list', type=str, required=True, help='input image or video input')
parser.add_argument('--dataset', type=str, required=True, help='dataset name')
parser.add_argument('--data_root', type=str, required=True, help='data root folder')
parser.add_argument('--ckpt', type=str, required=True, help='model ckpt path')
parser.add_argument('--output_folder', type=str, default='output', help='output video folder path')
parser.add_argument('--verbose', action='store_true', help='whether save results during generation for debug')
parser.add_argument('--clean', action='store_true', help='whether clean intermedia results afterwards')
parser.add_argument('--attach_lip', action='store_true', help='whether attach lip part only')

args = parser.parse_args()

face_detector = init_detection_model(model_name='retinaface_resnet50', half=True,
                                     device='cuda' if torch.cuda.is_available() else 'cpu')


def extract_frames(file_path, output_folder):
    streams = ffmpeg.input(file_path)
    streams.video.output(os.path.join(os.path.join(output_folder, 'frames'), f'%06d.{EXT}'),
                         **{'qscale:v': 0, 'r': FPS}).run(overwrite_output=True, quiet=True)
    streams.audio.output(os.path.join(output_folder, f'audio.wav'),
                         **{"qscale:a": 1, 'ar': 16000}).run(overwrite_output=True, quiet=True)


@torch.no_grad()
def face_crop(output_folder, face_detector):
    dataset = ImageFolder(os.path.join(output_folder, 'frames'), output_mode='cv2')

    # print(f"Total frame extracted {len(dataset)}")
    bsz = dataset.max_bsz_retinaface(0)
    dataloader = DataLoader(dataset, batch_size=bsz, num_workers=8, prefetch_factor=10)
    with open(os.path.join(output_folder, 'meta.txt'), 'w') as f:
        for data in tqdm(dataloader, total=len(dataloader), desc='face extraction', dynamic_ncols=True, position=2, leave=False):
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
                output_path = os.path.join(output_folder, 'crop_face', f'{name}.{EXT}')
                if args.dataset == 'lrs':
                    y1, y2, x1, x2 = 0, -40, 20, -20
                    cropped_face = img.numpy()[y1:y2, x1:x2, :]
                    cv2.imwrite(output_path, cropped_face)
                    bbox = x1, y1, x2, y2
                    f.write(meta_line(name, *bbox))
                    continue

                if bbox:
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
    input_file_list = []
    if args.dataset == 'lrs2':
        with open(args.input_list, 'r') as f:
            lines = f.readlines()
            for line in lines:
                input_file_list.append(os.path.join(f'{args.data_root}', line.strip('\n') + ".mp4"))
    else:
        NotImplementedError(f"{args.dataset} is not support yet.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask_module = Masking(half_precision=True, norm=False).to(device)
    model = lip_mage_vit_base(vq_config_path="config/vqgan.yml",
                              vq_state_dict="pretrained/vq_model_512_256.pt",
                              use_audio_reference=True,
                              use_image_reference=True,
                              tokenize_reference=True,
                              ref_control=True)

    model.load_state_dict(torch.load(args.ckpt))
    model.to(device)
    model.eval()

    for input_file in tqdm(input_file_list, desc=f'Evaluate {args.dataset}', dynamic_ncols=True, position=1):
        tmp_output_folder = os.path.join(TMP_FOLDER, '_'.join(os.path.splitext(input_file)[0].split('/')))

        if not os.path.exists(tmp_output_folder):
            os.makedirs(os.path.join(tmp_output_folder, 'frames'), exist_ok=True)
            os.makedirs(os.path.join(tmp_output_folder, 'crop_face'), exist_ok=True)
            os.makedirs(os.path.join(tmp_output_folder, 'align_face'), exist_ok=True)
            extract_frames(input_file, output_folder=tmp_output_folder)
            h, w = face_crop(output_folder=tmp_output_folder, face_detector=face_detector)
        else:
            frame_dir = os.path.join(tmp_output_folder, 'frames')
            img_file = [fname for fname in os.listdir(frame_dir) if fname.endswith(EXT)][0]
            h, w, _ = cv2.imread(os.path.join(frame_dir, img_file)).shape

        # print(f"Video Resolution {h}x{w}")

        wav = audio.load_wav(path=os.path.join(tmp_output_folder, 'audio.wav'), sr=SAMPLE_RATE)
        # print(f"Audio Length:{datetime.timedelta(seconds=len(wav) // SAMPLE_RATE)}")
        output_file = os.path.join(args.output_folder, input_file.replace(args.data_root, ''))
        os.makedirs(os.path.split(output_file)[0], exist_ok=True)
        # print(f"Output file: {output_file}")
        mel = audio.melspectrogram(wav).T
        dataset = GenerateDataset(tmp_output_folder, mel, dynamic_mask=True, mage=True)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                prefetch_factor=8,
                                num_workers=8)

        coords = dataset.coords

        process = (
            ffmpeg
            .input('pipe:', format='rawvideo',
                   pix_fmt='rgb24',
                   s='{}x{}'.format(w, h),
                   r=FPS,
                   thread_queue_size=1024)
            .output(ffmpeg.input(os.path.join(tmp_output_folder, 'audio.wav'), channel_layout="mono"),
                    output_file,
                    pix_fmt="yuv420p",
                    vcodec="libx264",
                    acodec='aac',
                    r=FPS,
                    crf=18,
                    )
            .overwrite_output()
            .run_async(pipe_stdin=True, quiet=True)
        )

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Processing {input_file}', dynamic_ncols=True,
                    leave=False, position=3)
        for i, (x, indiv_mels, ori_window, meta) in pbar:
            x = x.to(device, non_blocking=True)
            indiv_mels = indiv_mels.to(device, non_blocking=True)
            bsz = x.size(0)

            with torch.no_grad():
                x_masked = mask_module(x)
            masked_flag = mask_module.inverse_mask

            with torch.no_grad():
                with torch.autocast(device_type="cuda" if torch.cuda.is_available() else 'cpu',
                                    dtype=torch.float16 if torch.cuda.is_available() else torch.bfloat16,
                                    enabled=True):
                    (loss, acc), g, token_all_mask = model(x_masked,
                                                           gt=x,
                                                           ref=x,
                                                           audio=indiv_mels,
                                                           generate=True)
                g = g.to(torch.float32).clamp(-1, 1) / 2 + 0.5
            for batch_id, (face, frame, name) in enumerate(zip(g.unbind(0), ori_window.unbind(0), meta)):
                frame_idx = i * bsz + batch_id
                x1, y1, x2, y2 = coords[name]
                frame = frame.flip(-1).numpy()
                face = (face * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()

                ori_face = cv2.resize(frame[y1:y2, x1:x2], dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
                face_mask = masked_flag[batch_id, ...].squeeze().cpu().numpy().astype(np.float32)
                inv_mask_erosion = cv2.erode(face_mask, np.ones((2, 2), np.uint8))

                pasted_face = face
                total_face_area = np.sum(inv_mask_erosion)  # // 3
                w_edge = int(total_face_area ** 0.5) // 20
                erosion_radius = w_edge * 2
                inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
                blur_size = w_edge * 2
                inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)
                inv_soft_mask = inv_soft_mask[:, :, None]
                face = (inv_soft_mask * pasted_face + (1 - inv_soft_mask) * ori_face).astype(np.uint8)
                resize_face = cv2.resize(face, dsize=(x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)
                if args.verbose:
                    g = face.copy()
                    ref = cv2.resize(frame[y1:y2, x1:x2], (256, 256))
                    concat = np.flip(np.concatenate([ref, g], axis=1), axis=-1)
                    diff = np.abs(g.astype(float) - ref.astype(float))
                    diff /= diff.max()  # diff.mean(axis=-1, keepdims=True) * 255
                    os.makedirs(f"{tmp_output_folder}/diff", exist_ok=True)
                    os.makedirs(f"{tmp_output_folder}/compare", exist_ok=True)
                    os.makedirs(f"{tmp_output_folder}/mask", exist_ok=True)
                    cv2.imwrite(f'{tmp_output_folder}/diff/{frame_idx:06d}.jpg', np.abs(diff).astype(np.uint8))
                    cv2.imwrite(f'{tmp_output_folder}/compare/{frame_idx:06d}.jpg', concat.astype(np.uint8))
                    cv2.imwrite(f'{tmp_output_folder}/mask/{frame_idx:06d}.jpg', (face_mask * 255).astype(np.uint8))
                frame[y1:y2, x1:x2] = resize_face
                process.stdin.write(frame.astype(np.uint8).tobytes())

        process.stdin.close()
        process.wait()
