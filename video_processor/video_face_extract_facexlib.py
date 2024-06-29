import argparse
import ffmpeg
import os
import logging
import cv2
import traceback
# tarfile
import tarfile
from io import BytesIO
from functools import partial
from pathlib import Path
from glob import glob
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time
from tqdm import tqdm
from shutil import rmtree

import torch
from torch.utils.data import DataLoader
from torchvision.io import encode_jpeg

from facexlib.detection import init_detection_model
from utils import ImageFolder, get_largest_face, get_center_face, EMA

import h5py
import numpy as np
import warnings

warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True,
                    help='Dir containing youtube clips.')
parser.add_argument('--output_dir', type=str, required=True,
                    help='Location to dump outputs.')
parser.add_argument('--num_workers', type=int, default=4,
                    help='How many multiprocessing workers')
parser.add_argument('--max_frames', type=int, default=None,
                    help='Max frames extracted in total')
parser.add_argument('--resume_file', type=str, default=None,
                    help='resume processed file')
parser.add_argument('--failed_file', type=str, default=None,
                    help='only processed failed file')
parser.add_argument('--face_size', type=int, default=256,
                    help='face_size')
parser.add_argument('--cache', action='store_true',
                    help='whether cache file')
parser.add_argument('--tar', action='store_true',
                    help='whether tar file')
parser.add_argument('--raw', action='store_true',
                    help='whether raw byte file')
parser.add_argument('--original', action='store_true',
                    help='using original file')
parser.add_argument('--landmark_alignment', '-la',action='store_true',
                    help='landmark alignment')
parser.add_argument('--ext', type=str, default='jpg', choices=['jpg', 'png'],
                    help='Extension for image frames')
parser.add_argument('--name', type=str, default=str(int(time())),
                    help='name for logger file')
parser.add_argument('--min_frame', type=int, default=100,
                    help='minimum faces per videos')
parser.add_argument('--fps', type=int, default=25,
                    help='FPS')
parser.add_argument('--audio_path', type=str, default=None,
                    help='audio_path for dataset the split video & audio')
parser.add_argument('--audio_ext', type=str, default='wav',
                    help='audio extension for dataset the split video & audio')
parser.add_argument('--video_path', type=str, default=None,
                    help='video_path for dataset the split video & audio')
args = parser.parse_args()

EXT = args.ext
FPS = args.fps
audio_ext = args.audio_ext

logging.basicConfig(filename=f'extract_av_{args.name}.log', filemode='w', level=logging.INFO, format='%(asctime)s::%(levelname)s::%(lineno)d::%(message)s')

os.makedirs('processed', exist_ok=True)
record_logger = logging.getLogger("processed_file")
record_logger.setLevel(logging.INFO)
record_handler = logging.FileHandler(filename=f'processed/file_{args.name}.log', mode='w')
record_handler.setLevel(logging.INFO)
record_handler.setFormatter(logging.Formatter('%(message)s'))
record_logger.addHandler(record_handler)

os.makedirs('failed', exist_ok=True)
fail_logger = logging.getLogger("failed_file")
fail_logger.setLevel(logging.INFO)
fail_handler = logging.FileHandler(filename=f'failed/file_{args.name}.log', mode='w')
fail_handler.setLevel(logging.INFO)
fail_handler.setFormatter(logging.Formatter('%(message)s'))
fail_logger.addHandler(fail_handler)

os.makedirs('check', exist_ok=True)
check_logger = logging.getLogger("check_file")
check_logger.setLevel(logging.INFO)
check_handler = logging.FileHandler(filename=f'check/file_{args.name}.log', mode='w')
check_handler.setLevel(logging.INFO)
check_handler.setFormatter(logging.Formatter('%(message)s'))
check_logger.addHandler(check_handler)

os.makedirs('remove', exist_ok=True)
remove_logger = logging.getLogger("remove_file")
remove_logger.setLevel(logging.INFO)
remove_handler = logging.FileHandler(filename=f'remove/file_{args.name}.log', mode='w')
remove_handler.setLevel(logging.INFO)
remove_handler.setFormatter(logging.Formatter('%(message)s'))
remove_logger.addHandler(remove_handler)

if not args.original:
    face_detectors = [init_detection_model(model_name='retinaface_resnet50', half=True, device=f'cuda:{idx}') for idx in range(args.num_workers)]
face_template = np.array([[192.98138, 239.94708],
                          [318.90277, 240.1936],
                          [256.63416, 314.01935],
                          [201.26117, 371.41043],
                          [313.08905, 371.15118]])
face_size = (args.face_size, args.face_size)
face_template = face_template * (face_size[0] / 512.0)

def video_extract(file_path, output_dir, job_id, audio_path=None, video_path=None):

    if not os.path.isfile(file_path):
        logging.warning(f"{file_path} not found")
        return 
    
    if args.max_frames:
        current_files = len(glob(f"{output_dir}/**/*.{EXT}", recursive=True))
        logging.info(f"{file_path}: current files {current_files}/{args.max_frames}[{current_files/args.max_frames:.02%}]")
        if current_files > args.max_frames:
            return
    else:
        current_files = len(glob(f"{output_dir}/**/*.{EXT}", recursive=True))
        logging.info(f"{file_path}: current files {current_files}")
    pid = job_id


    filename = Path(file_path).stem
    
    try:
        probe = ffmpeg.probe(file_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        if audio_path is None:
            audio_info = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
            if not audio_info:
                logging.warning(f"{file_path} has No audio, skip.")
                return
        fps = eval(video_info['r_frame_rate'])
    except Exception as e:
        logging.error(f"{e.stderr} at probe {file_path}")
        fail_logger.info(file_path)

    output_folder = os.path.splitext(file_path.replace(args.input_dir, args.output_dir))[0]
    os.makedirs(output_folder, exist_ok=True)
    
    # streams = ffmpeg.input(file_path)
    logging.info(f"{output_folder} frame extraction")
    try:
        ffmpeg.input(file_path).video.output(os.path.join(output_folder, f'%06d.{EXT}'),
                                            **{'qscale:v':0, 'r': FPS}).run(overwrite_output=True, quiet=True)
    except ffmpeg.Error as e:
        logging.error("Error in frame extraction")
        logging.error(e.stderr.decode('utf-8'))
        fail_logger.info(file_path)
        return
    try:
        if video_path is not None and audio_path is not None:
            file_path = os.path.splitext(file_path.replace(video_path, audio_path))[0].rstrip('.') + "." + audio_ext.lstrip('.')
        ffmpeg.input(file_path).audio.output(os.path.join(output_folder, f'audio.wav'),
                                             **{"qscale:a": 1, 'ar': 16000}).run(overwrite_output=True, quiet=True)
    except ffmpeg.Error as e:
        logging.error("Error in audio extraction")
        logging.error(e.stderr.decode('utf-8'))
        fail_logger.info(file_path)
        return
    ema_landmark = EMA(decay=0.9)
    ema_bbox = EMA(decay=0.9)

    # create tarfile
    if args.tar:
        tf = tarfile.open(os.path.join(output_folder, 'data.tar'), 'w')

    try:
        if args.cache:
            logging.info(f"Caching images at {Path(output_folder) / 'cache.h5'}")
            h5_cache =  h5py.File(Path(output_folder) / 'cache.h5', 'w', libver='latest')

        logging.info(f"Write meta data to {Path(output_folder) / 'meta_data.txt'}")
        with open(Path(output_folder) / 'meta_data.txt', 'w') as meta_file:
            dataset = ImageFolder(output_folder, ext=EXT, output_mode='cv2')
            bsz = dataset.max_bsz_retinaface(pid) if not args.original else 32
            logging.info(f"BSZ {bsz} at {output_folder} for face crop")
            dataloader = DataLoader(dataset, batch_size=bsz, shuffle=False, num_workers=4 if not args.original else 0)
            if not args.original:
                face_detector = face_detectors[pid]
            all_landmarks = []
            ignore_file = False
            for data in dataloader:
                names, imgs = data['name'], data['img']
                if args.original:
                    for name, img in zip(names, imgs):
                        meta_file.write(name+'\n')
                        output_path = os.path.join(output_folder, f'{name}.{EXT}')     
                        img = encode_jpeg(img.permute(2, 0, 1), 100)
                        if args.cache:
                            h5_cache.create_dataset(
                                name, data=img, maxshape=img.shape,
                                compression='lzf', shuffle=True, track_times=False,)
                        os.remove(output_path)
                    continue
                try: 
                    with torch.no_grad():
                        batched_bboxes, batched_landmarks = face_detector.batched_detect_faces(imgs,
                                                                                               conf_threshold=0.95,
                                                                                               nms_threshold=0.3,
                                                                                               use_origin_size=True)
                except Exception as e:
                    logging.error(f"{output_folder} {e} at inference")
                    fail_logger.info(file_path)
                    return
                
                b, h, w, c = imgs.size()
                batched_det_faces = []
                batched_det_landmarks = []
                try:
                    for bboxes, landmarks in zip(batched_bboxes, batched_landmarks):
                        det_faces = []
                        det_landmarks = []
                        if len(bboxes) == 0:
                            batched_det_faces.append(None)
                            batched_det_landmarks.append(None)
                        else:
                            for bbox, landmark in zip(bboxes, landmarks):
                                det_faces.append(bbox[0:5])
                                det_landmarks.append(np.array(np.split(landmark, 5, axis=0)))
                            det_faces, face_idx = get_largest_face(det_faces, h=h, w=w)
                            batched_det_faces.append(det_faces.astype(int).tolist()[:4])
                            batched_det_landmarks.append(det_landmarks[face_idx])
                except Exception as e:
                    logging.error(f"failed at {output_path} at bbox extraction")
                else:
                    all_landmarks.extend(batched_det_landmarks)
                
                
                for name, bbox, landmark, img in zip(names, batched_det_faces, batched_det_landmarks, imgs):
                    output_path = os.path.join(output_folder, f'{name}.{EXT}')  # override original image
                    if ignore_file:
                        os.remove(output_path)
                        continue
                    if bbox:
                        if (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
                            ema_bbox.update(np.array(bbox))
                        # x1, y1, x2, y2 = bbox
                        if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]) and ema_bbox.avg_value is None:
                            logging.error(f"{output_path} Bad detection at first image {img.shape} {bbox}, remove folder")
                            try:
                                rmtree(output_folder)
                            except OSError as e:
                                logging.error(f'failed to remove {output_folder}, please remove afterwards.')
                                remove_logger.info(output_folder)
                            return
                        
                        x1, y1, x2, y2 = ema_bbox.avg_value.astype(int).tolist()
                        if args.landmark_alignment:
                            ema_landmark.update(landmark)
                            landmark = ema_landmark.avg_value
                    else:
                        if eval(name.lstrip('0')) < args.min_frame:
                            logging.error(f"{output_path} failed to find face. Remove folder {output_folder}")
                            try:
                                rmtree(output_folder)
                            except OSError as e:
                                logging.error(f'failed to remove {output_folder}, please remove afterwards.')
                                remove_logger.info(output_folder)
                            return
                        else:
                            logging.warning(f"{output_path} failed to find face. but got {args.min_frame}+ frames ignore unprocessed frames")
                            ignore_file = True
                            os.remove(output_path)
                            continue
                    try:
                        if args.landmark_alignment:
                            border_mode = cv2.BORDER_CONSTANT
                            affine_matrix = cv2.estimateAffinePartial2D(landmark, face_template, method=cv2.LMEDS)[0]
                            # inverse_affine = cv2.invertAffineTransform(affine_matrix)
                            cropped_face = cv2.warpAffine(img.numpy(), affine_matrix, face_size, borderMode=border_mode, borderValue=(135, 133, 132))
                            tensor_image = torch.tensor(cropped_face).permute(2, 0, 1).flip(0)
                        else:
                            cropped_face = img.numpy()[y1:y2, x1:x2, :]
                            tensor_image = img[y1:y2, x1:x2, :].permute(2, 0, 1).flip(0)
                        meta_file.write(name+'\n')                            
                        if args.cache:
                            tensor_image = encode_jpeg(tensor_image, 100)
                            h5_cache.create_dataset(
                                name, data=tensor_image, maxshape=tensor_image.shape,
                                compression='lzf', shuffle=True, track_times=False,)
                            os.remove(output_path)
                        else:
                            cv2.imwrite(output_path, cropped_face)
                    except Exception as e:
                        logging.error(f"{output_path} {e}")
                        fail_logger.info(file_path)
                        raise output_path
    except Exception as e:
        logging.error(f"{output_folder} {e}  at caching")
        fail_logger.info(file_path)
        if args.tar:
            tf.close()
        if args.cache:
            h5_cache.close()
        return
    else:
        if args.tar:
            tf.close()
        if args.cache:
            h5_cache.close()
        record_logger.info(file_path)
        logging.info(f"Caching {Path(output_folder)} finished")


def mp_handler(job):
    try:
        video_extract(*job)
    except KeyboardInterrupt:
        exit(0)
    except:
        traceback.print_exc()

if __name__ == '__main__':
    logging.info(args)
    if not args.input_dir.endswith('/'):
        args.input_dir += '/'
    if not args.output_dir.endswith('/'):
        args.output_dir += '/'
    os.makedirs(os.path.join(args.output_dir), exist_ok=True)
    if args.failed_file:
        filelist = []
        with open(args.failed_file, 'r') as f:
            for line in f.readlines():
                filelist.append(line.strip('\n'))
        logging.info(f"Only process from {args.failed_file}")
    else:
        filelist = sorted(glob(f"{args.input_dir}/**/*.mp4", recursive=True))

        if args.resume_file:
            previous_file = []
            with open(args.resume_file, 'r') as f:
                for line in f.readlines():
                    previous_file.append(line.strip('\n'))
            filelist = sorted(list(set(filelist).difference(set(previous_file))))
            logging.info(f"Resume from {args.resume_file}")

    print(f"#files to process {len(filelist)}")
    if args.num_workers == 1:
        for file_path in filelist:
            video_extract(file_path, args.output_dir, 0)
    else:
        if args.original:
            print("original")
            extract_func = partial(video_extract, output_dir=args.output_dir, job_id=0, audio_path=args.audio_path, video_path=args.video_path)
            with Pool(processes=args.num_workers) as pool:
                _ = list(tqdm(pool.imap_unordered(extract_func, filelist), total=len(filelist), dynamic_ncols=True, desc=args.name))
        else:
            jobs = [(fname, args.output_dir, i%args.num_workers, args.audio_path, args.video_path) for i, fname in enumerate(filelist)]
            p = ThreadPoolExecutor(args.num_workers)
            futures = [p.submit(mp_handler, j) for j in jobs]
            _ = [r.result() for r in tqdm(as_completed(futures),
                                        total=len(futures),
                                        dynamic_ncols=True,
                                        desc=args.name)]