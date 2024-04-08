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

face_template = np.array([[192.98138, 239.94708],
                          [318.90277, 240.1936],
                          [256.63416, 314.01935],
                          [201.26117, 371.41043],
                          [313.08905, 371.15118]]) / 512 * SIZE

meta_line = lambda name, bbox, lm, inv_affine: f"{name} {','.join(map(str, bbox))} " \
                                               f"{','.join(lm.flatten().astype(str).tolist())} " \
                                               f"{','.join(inv_affine.flatten().astype(str).tolist())}\n"

parser = argparse.ArgumentParser(description='lip-sync')
parser.add_argument('--input_list', type=str, required=True, help='input image or video input')
parser.add_argument('--dataset', type=str, required=True, help='dataset name')
parser.add_argument('--data_root', type=str, required=True, help='data root folder')
parser.add_argument('--ckpt', type=str, required=True, help='model ckpt path')
parser.add_argument('--output', type=str, default='output.mp4', help='output video path')
parser.add_argument('--verbose', action='store_true', help='whether save results during generation for debug')
parser.add_argument('--clean', action='store_true', help='whether clean intermedia results afterwards')
parser.add_argument('--attach_lip', action='store_true', help='whether attach lip part only')

args = parser.parse_args()

face_detector = init_detection_model(model_name='retinaface_resnet50', half=True,
                                     device='cuda' if torch.cuda.is_available() else 'cpu')


def extract_frames(file_path, output_folder):
    streams = ffmpeg.input(file_path)
    streams.video.output(os.path.join(output_folder, f'%06d.{EXT}'),
                         **{'qscale:v': 0, 'r': FPS}).run(overwrite_output=True, quiet=True)
    streams.audio.output(os.path.join(output_folder, f'audio.wav'),
                         **{"qscale:a": 1, 'ar': 16000}).run(overwrite_output=True, quiet=True)


@torch.no_grad()
def face_crop(output_folder):
    dataset = ImageFolder(os.path.join(output_folder, 'frames'), output_mode='cv2')
    print(f"Total frame extracted {len(dataset)}")
    bsz = dataset.max_bsz_retinaface(0)
    dataloader = DataLoader(dataset, batch_size=bsz, num_workers=8, prefetch_factor=10)
    with open(os.path.join(output_folder, 'meta.txt'), 'w') as f:
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
                output_path_crop = os.path.join(output_folder, 'crop_face', f'{name}.{EXT}')
                output_path_align = os.path.join(output_folder, 'align_face', f'{name}.{EXT}')

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
    # del face_detector
    return h, w


if __name__ == '__main__':
    input_file_list = []
    if args.dataset == 'lrs2':
        with open(args.input_list, 'r') as f:
            lines = f.readlines()
            for line in lines:
                input_file_list.append(os.path.join(f'{args.data_root}', line.strip('\n') + "/mp4"))
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
        TMP_FOLDER = os.path.join(TMP_FOLDER, '_'.join(os.path.splitext(input_file)[0].split('/')))
        os.makedirs(os.path.join(TMP_FOLDER, 'frames'), exist_ok=True)
        os.makedirs(os.path.join(TMP_FOLDER, 'crop_face'), exist_ok=True)
        os.makedirs(os.path.join(TMP_FOLDER, 'align_face'), exist_ok=True)
        if not os.path.exists(TMP_FOLDER):
            extract_frames(input_file, output_folder=os.path.join(TMP_FOLDER, 'frames'))
            h, w = face_crop(output_folder=TMP_FOLDER)
        else:
            frame_dir = os.path.join(TMP_FOLDER, 'frames')
            img_file = [fname for fname in os.listdir(frame_dir) if fname.endswith(EXT)][0]
            h, w, _ = cv2.imread(os.path.join(frame_dir, img_file)).shape

        print(f"Video Resolution {h}x{w}")

        wav = audio.load_wav(path=os.path.join(TMP_FOLDER, 'audio.wav'), sr=SAMPLE_RATE)
        print(f"Audio Length:{datetime.timedelta(seconds=len(wav) // SAMPLE_RATE)}")
        print(f"Output file: {args.output}")
        mel = audio.melspectrogram(wav).T
        batch_size = 16
        dataset = GenerateDataset(TMP_FOLDER,
                                  mel,
                                  dynamic_mask=True,
                                  landmark=True,
                                  mage=True)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                prefetch_factor=8,
                                num_workers=8)

        coords = dataset.coords
        landmarks = dataset.landmarks
        inv_affine_matrices = dataset.inv_affine_matrices

        process = (
            ffmpeg
            .input('pipe:', format='rawvideo',
                   pix_fmt='rgb24',
                   s='{}x{}'.format(w, h),
                   r=FPS,
                   thread_queue_size=1024)
            .output(ffmpeg.input(args.audio, channel_layout="mono"),
                    args.output,
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
                    leave=False, position=2)
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
                landmark = landmarks[name]
                inverse_matrix = inv_affine_matrices[name]
                frame = frame.flip(-1).numpy()
                restored_face = (face * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                inv_restored = cv2.warpAffine(restored_face, inverse_matrix, (w, h))
                if not args.attach_lip:
                    mask = np.ones([SIZE, SIZE], dtype=np.float32)
                else:
                    mask = masked_flag[batch_id, ...].squeeze().cpu().numpy().astype(np.float32)

                inv_mask = cv2.warpAffine(mask, inverse_matrix, (w, h))
                inv_mask_erosion = cv2.erode(inv_mask, np.ones((2, 2), np.uint8))

                pasted_face = inv_restored
                total_face_area = np.sum(inv_mask_erosion)
                w_edge = int(total_face_area ** 0.5) // 20
                erosion_radius = w_edge * 2
                inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
                blur_size = w_edge * 2
                inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)
                inv_soft_mask = inv_soft_mask[:, :, None]
                frame = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * frame
                process.stdin.write(frame.astype(np.uint8).tobytes())

        process.stdin.close()
        process.wait()
