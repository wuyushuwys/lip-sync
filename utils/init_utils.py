import os
import argparse
import functools
import random

import torch
import numpy as np

from torch import distributed as dist


def init_process(args):
    torch.cuda.empty_cache()

    # Enable cudnn Optimization for static network structure
    torch.backends.cudnn.benchmark = True
    # Enable tensor-core
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if 'SLURM_NPROCS' in os.environ:
        ngpus_per_node = torch.cuda.device_count()
        if ngpus_per_node == 1:
            args.world_size = int(os.environ['SLURM_NPROCS']) * ngpus_per_node
        else:
            args.world_size = int(os.environ['SLURM_NPROCS']) * ngpus_per_node
        args.distributed = args.world_size > 1
    else:
        args.distributed = torch.cuda.device_count() > 1

    if args.distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        if 'SLURM_NPROCS' in os.environ:
            ngpus_per_node = torch.cuda.device_count()
            args.world_size = int(os.environ['SLURM_NPROCS']) * ngpus_per_node
            args.rank = int(os.environ['SLURM_PROCID']) * ngpus_per_node + local_rank  # global rank

            node_list = os.environ['SLURM_NODELIST']
            args.node_list = node_list  # All node you are using
            args.job_id = os.environ["SLURM_JOB_ID"]  # get job id
            args.local_rank = local_rank
            # set environs
            os.environ['MASTER_ADDR'] = os.environ['SLURM_LAUNCH_NODE_IPADDR']  # get master addr
            os.environ['WORLD_SIZE'] = str(args.world_size)
            os.environ['LOCAL_RANK'] = str(local_rank)
            os.environ['RANK'] = str(args.rank)
        else:
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.rank = local_rank
            args.local_rank = local_rank
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)

        global_rank, world_size = get_dist_info()
        print(f"global rank: {global_rank}\tlocal rank: {local_rank}\tworld size: {world_size}")

    else:
        args.local_rank = 0
        args.rank = 0
        args.world_size = 1
        device = 0
        torch.cuda.set_device(device)
    # debug_mode(args)

    if args.manual_seed:
        torch.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        random.seed(args.manual_seed)


def get_dist_info():
    initialized = dist.is_initialized()
    world_size = dist.get_world_size() if initialized else 1
    rank = dist.get_rank() if initialized else 0
    return rank, world_size


def get_device(is_gpu=True):
    """Return the correct device"""
    rank, _ = get_dist_info()
    local_rank = rank % torch.cuda.device_count()
    return torch.device(local_rank if torch.cuda.is_available() and is_gpu else "cpu")


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
                # if isinstance(content, argparse.Namespace):
                if getattr(content, attr): return func(*args, **kwargs)
            for _, content in kwargs.items():
                # if isinstance(content, argparse.Namespace):
                if getattr(content, attr): return func(*args, **kwargs)

        return wrapper

    return decorator


# @when_attr_is_true('debug')
# @master_only
# def debug_mode(args: argparse.Namespace):
#     args.logger.info('Enable anomaly detect')
#     args.logger.warning('Debug mode is super slow, set epochs to 1')
#     args.epochs = 1
#     torch.autograd.set_detect_anomaly(args.debug)
