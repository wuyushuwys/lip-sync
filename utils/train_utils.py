import argparse
import importlib
import os

from typing import Optional, Dict, Union

import torch
import math

from torch.utils.data import ConcatDataset

from collections import OrderedDict
from pathlib import Path

from torch.utils.data import DataLoader
from torch.distributed.optim import ZeroRedundancyOptimizer

from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig

import utils
from utils import lr_scheduler, gradual_warmup_scheduler
from utils.prefetch_dataloader import CUDAPrefetcher
from utils.init_utils import master_only, get_dist_info
from utils.logging_tool import get_logger

__all__ = ["create_dataloader",
           "create_criteria",
           "load_ckpt",
           "state_dict_saver",
           "ckpt_loader",
           'ckpt_saver',
           "create_optim_scheduler",
           ]


def create_dataloader(args):
    logger = get_logger()
    dataset_modules = [importlib.import_module(f'datasets.{dataset}') for dataset in args.dataset]
    train_dataset = ConcatDataset([module.get_dataset(utils.mode.TRAIN, args) for module in dataset_modules])
    logger.info(f"Total training data:{len(train_dataset)}")
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=args.rank,
                                                                    shuffle=True) if args.distributed else None

    # Load eval dataset
    if args.eval_datasets:
        eval_datasets = []
        eval_samplers = dict()
        for eval_dataset in args.eval_datasets:
            eval_dataset_module = importlib.import_module(f'datasets.{eval_dataset}')
            eval_datasets.append((eval_dataset, eval_dataset_module.get_dataset(utils.mode.EVAL, args)))
            eval_samplers[eval_dataset] = torch.utils.data.distributed.DistributedSampler(
                eval_dataset) if args.distributed else None
    else:

        dataset_modules = [importlib.import_module(f'datasets.{dataset}') for dataset in args.dataset]
        eval_datasets = [module.get_dataset(utils.mode.EVAL, args) for module in dataset_modules]
        eval_datasets = [(name, eval_dataset) for name, eval_dataset in zip(args.dataset, eval_datasets)]
        eval_samplers = {}
        for name, dataset in eval_datasets:
            eval_samplers[name] = torch.utils.data.distributed.DistributedSampler(dataset) if args.distributed else None

    prefetch_factor = args.data_spec['prefetch_factor'] if 'prefetch_factor' in args.data_spec.keys() else 2

    _, world_size = get_dist_info()
    logger.info(f'Effective batch_size: {args.batch_size * world_size}')
    # Dataloader
    train_data_loader = DataLoader(dataset=train_dataset,
                                   num_workers=args.num_workers,
                                   batch_size=args.batch_size,
                                   shuffle=(train_sampler is None),
                                   drop_last=True,
                                   pin_memory=True,
                                   sampler=train_sampler,
                                   prefetch_factor=prefetch_factor)

    # train_data_loader = CUDAPrefetcher(loader=train_data_loader, args=args)

    eval_kwargs = {"num_workers": args.num_workers,
                   "batch_size": args.batch_size,
                   'shuffle': False,
                   'drop_last': False,
                   'pin_memory': True}
    eval_data_loaders = [(data_name, DataLoader(dataset=dataset,
                                                sampler=eval_samplers[data_name], **eval_kwargs)) for
                         data_name, dataset in eval_datasets]

    args.total_iterations = int(args.epochs * len(train_data_loader))

    if args.log_steps == 0:
        args.log_steps = max(min(len(train_data_loader) // 10, 100), 1)
    return train_data_loader, train_sampler, eval_data_loaders, eval_samplers


def subdict(dict: Union[Dict, DictConfig], *exceptions) -> Dict:
    """
    Return a dictionary exclude specific keys without modifying original dict
    Args:
        dict: dictionary
        *exceptions: keys that exclude

    Returns: Sub-Dictionary

    """
    return {k: v for k, v in dict.items() if k not in exceptions}


def create_criteria(args: argparse.Namespace):
    assert hasattr(args, 'losses'), "Missing losses in model config"
    losses_module = importlib.import_module("losses")
    criteria = OrderedDict()
    logger = get_logger()
    if args.losses is not None:
        for name, kwargs in args.losses.items():
            if 'type' in kwargs:
                logger.info(f"Create Loss[{name}]:{kwargs}")
                loss = getattr(losses_module, kwargs.get('type'))
                criteria[name] = loss(**subdict(kwargs, 'type')).to(args.local_rank)

        return criteria
    else:
        logger.info('No criterion initialized')
        return None


def create_optim_scheduler(*model_list: [torch.nn.Module], args: argparse.Namespace, num_batches: int):
    logger = get_logger()

    def _get_optim(optim_type):
        if optim_type == 'SM3':
            try:
                from SM3 import SM3
            except ImportError:
                raise ImportError("No SM3 found. Please install SM3 via pip install torch-SM3")
            return SM3
        else:
            return getattr(torch.optim, optim_type)

    assert hasattr(args, 'optim'), "Missing optim in model config"
    if isinstance(args.optim, ListConfig):
        optim_module = [_get_optim(optim_args.get('type')) for optim_args in args.optim]
        optim_dict = [subdict(optim_args, 'type') for optim_args in args.optim]
        if args.distributed and args.scale_lr:
            for optim_arg in optim_dict:
                original_lr = optim_arg.get('lr')
                scalar = args.world_size
                optim_arg['lr'] *= scalar
                logger.info(f"Scale learning rate from {original_lr} --> {optim_arg.get('lr')}")
    elif isinstance(args.optim, DictConfig):
        optim_module = _get_optim(args.optim.get('type'))
        optim_dict = subdict(args.optim, 'type')
        if args.distributed and args.scale_lr:
            original_lr = optim_dict.get('lr')
            scalar = args.world_size
            optim_dict['lr'] *= scalar
            logger.info(f"Scale learning rate from {original_lr} --> {optim_dict.get('lr')}")
    else:
        raise NotImplementedError(type(args.optim))

    total_iters = args.epochs * num_batches

    if args.warmup_lr:
        if hasattr(args, 'warmup_iters'):
            warmup_iters = int(total_iters * args.warmup_iters) if args.warmup_iters < 1 else args.warmup_iters
        else:
            logger.info('warmup_iters not set, warmup for 1% iterations')
            warmup_iters = total_iters // 100
        logger.info(f"Warm up lr {warmup_iters}/{total_iters} iterations")
        total_iters -= warmup_iters

    assert hasattr(args, 'scheduler'), "Missing scheduler in model config"

    scheduler_module = getattr(lr_scheduler, args.scheduler.get('type'))
    if issubclass(scheduler_module, torch.optim.lr_scheduler.MultiStepLR):
        args.scheduler['milestones'] = [int(math.ceil(total_iters * i)) for i in args.scheduler['milestones']]
    elif issubclass(scheduler_module, lr_scheduler.CosineAnnealingRestartLR):
        # evenly divide periods
        args.scheduler['periods'] = [total_iters // len(args.scheduler['restart_weights']) for _ in
                                     range(len(args.scheduler['restart_weights']))]

        # periods base on percentage e.g. epochs=40 [0, 0.25, 0.50, 0.75, 1] --> [10, 10, 10, 10]
        # args.scheduler['periods'] = [
        #     num_iters * args.epochs * (args.scheduler['periods'][idx + 1] - args.scheduler['periods'][id]) for idx in
        #     range(len(args.scheduler['periods']) - 1)
        # ]
    elif issubclass(scheduler_module, torch.optim.lr_scheduler.CosineAnnealingLR):
        args.scheduler["T_max"] = total_iters
    else:
        args.scheduler['total_iters'] = total_iters

    optimizer_list = []
    scheduler_list = []
    if isinstance(args.optim, ListConfig):
        logger.info(f'Create {len(optim_dict)} optim configs for {len(model_list)} models')
        for model, optim, optim_args in zip(model_list, optim_module, optim_dict):
            # todo: modify if needed

            params_group = filter(lambda p: p.requires_grad, model.parameters())

            optimizer = optim(params_group, **optim_args)

            scheduler = scheduler_module(optimizer, **subdict(args.scheduler, 'type'))

            if args.warmup_lr:
                scheduler = gradual_warmup_scheduler.GradualWarmupScheduler(optimizer=optimizer,
                                                                            multiplier=1,
                                                                            total_epoch=warmup_iters,
                                                                            after_scheduler=scheduler)
            optimizer_list.append(optimizer)
            scheduler_list.append(scheduler)
    else:
        logger.info(f'Create same optim configs for {len(model_list)} models')
        for model in model_list:
            # todo: modify if needed
            if optim_dict.get('weight_decay', False):
                decay = []
                no_decay = []
                for name, p in model.named_parameters():
                    if not p.requires_grad:
                        continue
                    if p.ndim <= 1 or name.endswith(".bias") or name:
                        no_decay.append(p)
                    else:
                        decay.append(p)
                params_group = [
                    {'params': no_decay, 'weight_decay': 0.},
                    {'params': decay, 'weight_decay': optim_dict.pop("weight_decay")}]
            else:
                params_group = filter(lambda p: p.requires_grad, model.parameters())

            optimizer = optim_module(params_group, **optim_dict)

            scheduler = scheduler_module(optimizer, **subdict(args.scheduler, 'type'))

            if args.warmup_lr:
                scheduler = gradual_warmup_scheduler.GradualWarmupScheduler(optimizer=optimizer,
                                                                            multiplier=1,
                                                                            total_epoch=warmup_iters,
                                                                            after_scheduler=scheduler)
            optimizer_list.append(optimizer)
            scheduler_list.append(scheduler)

    return optimizer_list, scheduler_list


def _create_optim_scheduler(model: torch.nn.Module,
                            optim_module: torch.optim,
                            optim_args: dict,
                            scheduler_module,
                            args: argparse.Namespace,
                            warmup_iters: int = 0):
    optimizer = optim_module(filter(lambda p: p.requires_grad, model.parameters()),
                             **optim_args)
    scheduler = scheduler_module(optimizer, **subdict(args.scheduler, 'type'))

    if args.warmup_lr:
        scheduler = gradual_warmup_scheduler.GradualWarmupScheduler(optimizer=optimizer,
                                                                    multiplier=1,
                                                                    total_epoch=warmup_iters,
                                                                    after_scheduler=scheduler)
    return optimizer, scheduler


def load_ckpt(ckpt, **kwargs):
    for k, v in kwargs.items():
        if hasattr(v, 'load_state_dict'):
            v.load_state_dict(ckpt[k])


# @master_only  # not using master_only for FSDP support
def state_dict_saver(path, model):
    state_dict = model.state_dict() if not hasattr(model, 'module') else model.module.state_dict()
    if get_dist_info()[0] == 0:
        dir_checker(path)
        torch.save(state_dict, path)


# @master_only  # not using master_only for FSDP support
def ckpt_saver(path, **kwargs):

    ckpt = {}
    for k, v in kwargs.items():
        if isinstance(v, int):
            pass
        elif isinstance(v, torch.optim.Optimizer):
            v.zero_grad()  # clean grad in optimizer
        elif hasattr(v, 'module'):
            v = v.module
        if hasattr(v, 'state_dict'):
            v = v.state_dict()
        ckpt[k] = v
    if get_dist_info()[0] == 0:
        dir_checker(path)
        torch.save(ckpt, path)


def ckpt_loader(ckpt, **kwargs):
    for k, module in ckpt.items():
        if isinstance(module, int):
            pass
        else:
            if hasattr(kwargs[k], 'module'):
                kwargs[k].module.load_state_dict(ckpt[k])
            else:
                kwargs[k].load_state_dict(ckpt[k])


def dir_checker(file_path):
    path = Path(file_path).parent
    os.makedirs(path, exist_ok=True)


if __name__ == "__main__":
    pass
