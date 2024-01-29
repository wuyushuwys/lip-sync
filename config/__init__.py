import argparse
import yaml
import os

from omegaconf import OmegaConf
from pathlib import Path


def update_params(args: argparse.Namespace):
    if os.path.isfile(args.config_file):
        file_path = args.config_file
    else:
        file_path = Path(__file__).parent / args.config_file

    args = {k: v for k, v in vars(args).items() if not k.startswith('__')}
    with open(file_path, 'r') as f:
        args.update(yaml.safe_load(f.read()))
        args = OmegaConf.create(args)

    # save current config in yml
    os.makedirs(args.job_dir, exist_ok=True)
    if args.rank == 0:
        OmegaConf.save(args, os.path.join(args.job_dir, 'config.yml'))
    return args


if __name__ == "__main__":
    pass
