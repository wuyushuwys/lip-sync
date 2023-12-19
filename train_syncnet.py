import warnings

import argparse
import os

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb

import config
import common

from utils import *
from utils.logging_tool import get_logger
from models.syncnet import SyncNet


def train(model, optimizer, scheduler, criterion,
          train_data_loader, epoch, writer, args):
    logger = get_logger(args.job_dir)
    time_meter = common.meters.TimeMeter()
    losses_meter = common.meters.LossesMeter(fmt='.04e')

    model.train()
    nb = len(train_data_loader)
    log_vars = {'@loss': None, 'lr': None}
    batch_idx = 0
    train_data_loader.reset()
    batch = train_data_loader.next()
    while batch is not None:
        batch_idx += 1

        total_batches = (epoch - 1) * nb + batch_idx

        x, mel, y = batch

        optimizer.zero_grad()

        a, v = model(mel, x)

        loss = criterion['sync_loss'](a, v, y)
        loss.backward()

        log_vars['sync_loss'] = loss.item()
        log_vars['lr'] = scheduler.get_last_lr()[0]
        log_vars['@loss'] = loss.item()

        optimizer.step()
        scheduler.step()

        time_meter.update()
        losses_meter.update(log_vars, x.size(0))
        batch = train_data_loader.next()

        if batch_idx % args.log_steps == 0:
            time_meter.complete_time(nb - batch_idx)
            tb_writer(writer=writer, loss_dict=log_vars, nb=total_batches, tag='train')
            s = f"## Epoch:{epoch:{' '}{'>'}{2}d}/{params.epochs}\t" \
                f"Iters:{batch_idx:{' '}{'>'}{len(str(nb))}d}/{nb:d}({batch_idx / nb:.02%})\t" \
                f"Epoch-est. {time_meter.remain_time} {loss_printer(log_vars, fmt='.04e')}"
            logger.info(s)
    logger.info(f"Epoch{epoch:{' '}{'>'}{2}d}/{args.epochs} finished. SyncLoss: {losses_meter.avg}")


def main(args):
    logger = get_logger(args.job_dir)
    device = args.local_rank

    # Create job and tb_writer
    writer = SummaryWriter(args.job_dir) if args.rank == 0 else None
    # init wandb
    if args.rank == 0:
        wandb.init(project='lip-sync', dir=args.job_dir, name=args.job_dir.split('/')[-1], config=args)
    # Load train datasetcd
    train_data_loader, train_sampler, eval_data_loaders, eval_sampler = create_dataloader(args)

    # Create generator
    model = SyncNet()

    # profile_model(params)

    # Loss function
    criterion = create_criterions(args)

    # create optimizers and schedulers
    [optimizer], [scheduler] = create_optim_scheduler(model, args=args, num_batches=len(train_data_loader))

    # Load ckpt
    if args.resume and args.ckpt:
        pass
    else:
        start_epoch = 0

    # Load state_dict

    if args.weight:
        ckpt = torch.load(args.weight)
        model.load_state_dict(ckpt)
        logger.info(f"Load weight from {args.weight}")

    # allocate model to gpu
    if args.distributed:
        logger.info("Distributed Training")
        model = DDP(model.to(device), device_ids=[device], output_device=device)
    else:
        model.to(device)

    logger.info(attr_extractor(args))

    for epoch in range(start_epoch + 1, args.epochs + 1):
        # Train
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train(model, optimizer, scheduler, criterion, train_data_loader, epoch, writer, args)
        # Eval model
        evaluate = model.evaluate if hasattr(model, 'evaluate') else model.module.evaluate
        evaluate(model=model, eval_data_loaders=eval_data_loaders, epoch=epoch, writer=writer, args=args)
        # save model weight
        state_dict_saver(os.path.join(params.job_dir, 'weights', f'{args.model}.pt'), model)
        ckpt_saver(os.path.join(params.job_dir, "ckpt", f"{args.model}_latest.pth"),
                   model=model,
                   optimizer=optimizer, scheduler=scheduler,
                   epoch=epoch)

    logger.info(f"Finish Training")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arguments_parser(parser)

    # Parse arguments
    params = parser.parse_args()
    logger = get_logger(file_path=params.job_dir)
    params.logger = logger
    init_process(params)

    # read from config file
    config.update_params(params)

    main(params)
