import argparse
import logging
import os
import numpy as np
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool

from common.io import Hdf5
from utils.audio import load_wav, melspectrogram
from utils.hparams import hparams


def load_audio_melspec(file_name):
    wav = load_wav(path=file_name, sr=hparams.sample_rate)
    return melspectrogram(wav).T.astype(np.float32)


def process_audio(filename):
    data = load_audio_melspec(filename)
    np.save(os.path.splitext(filename)[0], data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Audio to Hdf5 Cache")
    parser.add_argument("--input-dir", type=str, required=True, help='Input path for dataset root')
    parser.add_argument("--output-dir", type=str, required=True, help='Output path for dataset root')
    parser.add_argument("--name", type=str, required=True, help='Name for hdf5 file {name}.h5')
    parser.add_argument('--ext', type=str, choices=['wav', 'mp3'], default='wav', help='audio extension')
    parser.add_argument('--npy', action='store_true', help='store in npy cache per files')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.info(
        f"Create cache {os.path.join(args.output_dir, f'{args.name}.h5')} from {args.input_dir}/**/*.{args.ext}")

    audio_list = sorted(glob(os.path.join(args.input_dir, f"**/*.{args.ext}"), recursive=True))
    logging.info(f"Trying to cache {len(audio_list)} audio")
    os.makedirs(args.output_dir, exist_ok=True)
    if not args.npy:
        dataset = Hdf5(fname=os.path.join(args.output_dir, f"{args.name}_sr_{int(hparams.sample_rate)}.h5"),
                       overwrite=True)

        for audio_file in tqdm(audio_list, dynamic_ncols=True):
            audio_data = load_audio_melspec(audio_file)
            dataset.add(audio_file, audio_data)
    else:
        with Pool(processes=32) as p:
            _ = list(tqdm(p.imap_unordered(process_audio, audio_list), total=len(audio_list), dynamic_ncols=True))

    logging.info("Cache complete")
