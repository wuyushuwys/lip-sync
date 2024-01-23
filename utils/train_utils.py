import argparse
import importlib
import os.path

import torch
import math

from torch.utils.data import ConcatDataset

from collections import OrderedDict
from pathlib import Path

from torch.utils.data import DataLoader

import utils
from utils import lr_scheduler, gradual_warmup_scheduler
from utils.prefetch_dataloader import CUDAPrefetcher
from utils.init_utils import master_only
from utils.logging_tool import get_logger

__all__ = ["create_dataloader",
           "create_criterions",
           "load_ckpt",
           "state_dict_saver",
           "ckpt_loader",
           'ckpt_saver',
           "create_optim_scheduler",
           ]


def create_dataloader(args):
    dataset_modules = [importlib.import_module(f'datasets.{dataset}') for dataset in args.dataset]
    train_dataset = ConcatDataset([module.get_dataset(utils.mode.TRAIN, args) for module in dataset_modules])

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

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    prefetch_factor = args.data_spec['prefetch_factor'] if 'prefetch_factor' in args.data_spec.keys() else 2
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


def subdict(dict: dict, *exceptions) -> dict:
    return {k: v for k, v in dict.items() if k not in exceptions}


def create_criterions(args: argparse.Namespace):
    # assert isinstance(args, argparse.Namespace), 'args should be an argparse.Namespace object'
    assert hasattr(args, 'losses'), "Missing losses in model config"
    # assert isinstance(args.losses, dict), "Losses in model config should be a dictionary"
    losses_module = importlib.import_module("losses")
    criterions = OrderedDict()
    for name, kwargs in args.losses.items():
        loss = getattr(losses_module, kwargs.get('type'))
        criterions[name] = loss(**subdict(kwargs, 'type')).to(args.local_rank)

    return criterions


def create_optim_scheduler(*model_list: [torch.nn.Module], args: argparse.Namespace, num_batches: int):
    # if not isinstance(model_list, list):
    #     model_list = [model_list]
    logger = get_logger()
    # assert isinstance(args, argparse.Namespace), 'args should be an argparse.Namespace object'
    assert hasattr(args, 'optim'), "Missing optim in model config"
    # assert isinstance(args.optim, dict), "optim in model config should be a dictionary"

    optim_module = getattr(torch.optim, args.optim.get('type'))
    optim_dict = subdict(args.optim, 'type')

    if args.distributed and args.scale_lr:
        original_lr = optim_dict.get('lr')
        scalar = args.world_size
        optim_dict['lr'] *= scalar

        logger.info(f"Scale learning rate from {original_lr}x{int(scalar):d} --> {optim_dict.get('lr')}")

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
    # assert isinstance(args.scheduler, dict), "scheduler in config should be a dictionary"

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
        NotImplementedError(f"Method {scheduler_module.__class__} is not implemented")
    optimizer_list = []
    scheduler_list = []
    for model in model_list:
        # todo: modify if needed
        optimizer = optim_module(filter(lambda p: p.requires_grad, model.parameters()),
                                 **optim_dict)
        scheduler = scheduler_module(optimizer, **subdict(args.scheduler, 'type'))

        if args.warmup_lr:
            scheduler = gradual_warmup_scheduler.GradualWarmupScheduler(optimizer=optimizer,
                                                                        multiplier=1,
                                                                        total_epoch=warmup_iters,
                                                                        after_scheduler=scheduler)
        optimizer_list.append(optimizer)
        scheduler_list.append(scheduler)

    return optimizer_list, scheduler_list


def load_ckpt(ckpt, **kwargs):
    for k, v in kwargs.items():
        if hasattr(v, 'load_state_dict'):
            v.load_state_dict(ckpt[k])


@master_only
def state_dict_saver(path, model):
    dir_checker(path)
    state_dict = model.state_dict() if not hasattr(model, 'module') else model.module.state_dict()
    torch.save(state_dict, path)


@master_only
def ckpt_saver(path, **kwargs):
    dir_checker(path)
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
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    pass
