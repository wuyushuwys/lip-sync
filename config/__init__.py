import argparse
import importlib
import yaml
import os
from pathlib import Path

PATH = os.path.dirname(os.path.realpath(__file__))


def update_params(args: argparse.Namespace):
    with open(Path(__file__).parent / args.config_file, 'r') as f:
        config = yaml.safe_load(f.read())
        for n, v in config.items():
            args.__setattr__(n, v)


if __name__ == "__main__":
    pass
