import argparse
import importlib
import itertools

import torch
import math

from collections import OrderedDict

import common

from torch.utils.data import DataLoader

from utils import lr_scheduler, gradual_warmup_scheduler
from utils.prefetch_dataloader import CUDAPrefetcher
from utils.init_utils import master_only, when_attr_is_true
from utils.helpers import default

__all__ = ["create_dataloader",
           "create_criterions",
           "load_ckpt",
           "state_dict_saver",
           'ckpt_saver',
           "create_optim_scheduler",
           # "clip_gradient"
           ]


def create_dataloader(args):
    dataset_module = importlib.import_module(f'datasets.{args.dataset}' if args.dataset else 'datasets')
    train_dataset = dataset_module.get_dataset(common.modes.TRAIN, args)

    # Load eval dataset
    if args.eval_datasets:
        eval_datasets = []
        for eval_dataset in args.eval_datasets:
            eval_dataset_module = importlib.import_module(f'datasets.{eval_dataset}')
            eval_datasets.append((eval_dataset, eval_dataset_module.get_dataset(common.modes.EVAL, args)))
    else:
        eval_datasets = [(args.dataset, dataset_module.get_dataset(common.modes.EVAL, args))]

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    eval_sampler = None
    prefetch_factor = 4
    # Dataloader
    train_data_loader = DataLoader(dataset=train_dataset,
                                   num_workers=args.num_data_threads,
                                   batch_size=args.train_batch_size,
                                   shuffle=(train_sampler is None),
                                   drop_last=True,
                                   pin_memory=True,
                                   sampler=train_sampler,
                                   prefetch_factor=prefetch_factor)

    train_data_loader = CUDAPrefetcher(loader=train_data_loader, args=args)

    eval_kwargs = {"num_workers": args.num_data_threads,
                   "batch_size": args.eval_batch_size,
                   'shuffle': False,
                   'drop_last': False,
                   'pin_memory': True,
                   'sampler': eval_sampler}
    eval_data_loaders = [(data_name, DataLoader(dataset=dataset, **eval_kwargs)) for data_name, dataset in
                         eval_datasets]

    args.total_iterations = int(args.epochs * len(train_data_loader))

    if args.log_steps == 0:
        args.log_steps = min(len(train_data_loader) // 100, 100)
    return train_data_loader, train_sampler, eval_data_loaders, eval_sampler


def subdict(dict: dict, *exceptions) -> dict:
    # exceptions = exceptions.split(',')
    return {k: v for k, v in dict.items() if k not in exceptions}


def create_criterions(args: argparse.Namespace):
    assert isinstance(args, argparse.Namespace), 'args should be an argparse.Namespace object'
    assert hasattr(args, 'losses'), "Missing losses in model config"
    assert isinstance(args.losses, dict), "Losses in model config should be a dictionary"
    losses_module = importlib.import_module("losses")
    criterions = OrderedDict()
    for name, kwargs in args.losses.items():
        loss = getattr(losses_module, kwargs.get('type'))
        if kwargs.get('type') == 'BitPerPixelLoss':
            kwargs['lambda_schedule']['steps'][0] *= args.total_iterations
            kwargs['target_schedule']['steps'][0] *= args.total_iterations
        criterions[name] = loss(**subdict(kwargs, 'type')).to(args.local_rank)

    return criterions


def create_optim_scheduler(model_list: [torch.nn.Module], args: argparse.Namespace, num_iters: float):
    if not isinstance(model_list, list):
        model_list = [model_list]
    assert isinstance(args, argparse.Namespace), 'args should be an argparse.Namespace object'
    assert hasattr(args, 'optim'), "Missing optim in model config"
    assert isinstance(args.optim, dict), "optim in model config should be a dictionary"

    optim_module = getattr(torch.optim, args.optim.get('type'))
    optim_dict = subdict(args.optim, 'type')

    if args.distributed:
        original_lr = optim_dict.get('lr')
        scalar = args.world_size
        optim_dict['lr'] *= scalar
        args.logger.info(f"Scale learning from {original_lr}X{int(scalar):d}==> {optim_dict.get('lr')}")

    total_iters = args.epochs * num_iters

    if args.warmup_lr:
        if hasattr(args, 'warmup_iters'):
            warmup_iters = int(total_iters * args.warmup_iters) if args.warmup_iters < 1 else args.warmup_iters
        else:
            args.logger.info('warmup_iters not set, warmup for 1% iterations')
            warmup_iters = total_iters // 100
        args.logger.info(f"Warm up lr {warmup_iters}/{total_iters} iterations")
        total_iters -= warmup_iters

    assert hasattr(args, 'scheduler'), "Missing scheduler in model config"
    assert isinstance(args.scheduler, dict), "scheduler in config should be a dictionary"

    scheduler_module = getattr(lr_scheduler, args.scheduler.get('type'))
    if isinstance(scheduler_module, torch.optim.lr_scheduler.MultiStepLR):
        args.scheduler['milestones'] = [int(math.ceil(total_iters * i)) for i in args.scheduler['milestones']]
    elif isinstance(scheduler_module, lr_scheduler.CosineAnnealingRestartLR):
        # evenly divide periods
        args.scheduler['periods'] = [total_iters // len(args.scheduler['restart_weights']) for _ in
                                     range(len(args.scheduler['restart_weights']))]

        # periods base on percentage e.g. epochs=40 [0, 0.25, 0.50, 0.75, 1] --> [10, 10, 10, 10]
        # args.scheduler['periods'] = [
        #     num_iters * args.epochs * (args.scheduler['periods'][idx + 1] - args.scheduler['periods'][id]) for idx in
        #     range(len(args.scheduler['periods']) - 1)
        # ]
    elif isinstance(scheduler_module, torch.optim.lr_scheduler.CosineAnnealingLR):
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
    state_dict = model.state_dict() if not hasattr(model, 'module') else model.module.state_dict()
    torch.save(state_dict, path)


@master_only
def ckpt_saver(path, **kwargs):
    ckpt = {}
    for k, v in kwargs.items():
        if isinstance(v, int):
            pass
        elif hasattr(v, 'module'):
            v = v.module
        if hasattr(v, 'state_dict'):
            v = v.state_dict()
        ckpt[k] = v
    torch.save(ckpt, path)


if __name__ == "__main__":
    pass
    # test_params = argparse.Namespace()
