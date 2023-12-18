import os
import threading
import time
import argparse
import functools

import torch
import numpy as np

from torch import distributed as dist


def init_process(args):
    torch.cuda.empty_cache()
    args.distributed = torch.cuda.device_count() > 1
    if args.distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        if 'SLURM_NPROCS' in os.environ:
            ngpus_per_node = torch.cuda.device_count()
            args.world_size = int(os.environ['SLURM_NPROCS']) * ngpus_per_node  # compute world_size for multi-node
            os.environ['MASTER_ADDR'] = os.environ['SLURM_LAUNCH_NODE_IPADDR']  # get master addr
            args.rank = int(os.environ['SLURM_PROCID']) * ngpus_per_node + local_rank  # global rank
            args.node_list = os.environ["SLURM_NODELIST"]  # All node you are using
            args.job_id = os.environ["SLURM_JOB_ID"]  # get job id
            args.local_rank = local_rank
        else:
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.rank = local_rank
            args.local_rank = local_rank
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)
        print(f"Init rank: {args.rank}\t local rank: {local_rank}")

        # dist.barrier(device_ids=[device])
    else:
        args.local_rank = 0
        args.rank = 0
        args.world_size = 1
        device = 0
        torch.cuda.set_device(device)
    debug_mode(args)

    if args.manual_seed:
        torch.manual_seed(args.rank)
        np.random.seed(args.rank)


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def get_device(is_gpu=True):
    """Return the correct device"""
    rank, _ = get_dist_info()
    local_rank = rank % torch.cuda.device_count()
    return torch.device(local_rank if torch.cuda.is_available() and is_gpu
                        else "cpu")


def master_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper


def when_attr_is_true(attr):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for content in args:
                if isinstance(content, argparse.Namespace):
                    if getattr(content, attr): return func(*args, **kwargs)
            for _, content in kwargs.items():
                if isinstance(content, argparse.Namespace):
                    if getattr(content, attr): return func(*args, **kwargs)

        return wrapper

    return decorator


@when_attr_is_true('debug')
@master_only
def debug_mode(args: argparse.Namespace):
    args.logger.info('Enable anomaly detect')
    args.logger.warning('Debug mode is super slow, set epochs to 1')
    args.epochs = 1
    torch.autograd.set_detect_anomaly(args.debug)
