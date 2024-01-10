import argparse
import os

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb

import config
import common

from utils.args_parser import arguments_parser
from utils.init_utils import init_process
from utils.train_utils import (create_dataloader, create_criterions, create_optim_scheduler,
                               ckpt_saver, state_dict_saver, ckpt_loader)
from utils.logger_utils import tb_writer, loss_printer, attr_extractor
from utils.logging_tool import get_logger

from models.wav2lip import Wav2Lip


def train(model, optimizer, scheduler, criterion,
          train_data_loader, epoch, writer, args, logger):
    time_meter = common.meters.TimeMeter()
    losses_meter = common.meters.LossesMeter(fmt='.04e')

    model.train()
    nb = len(train_data_loader)
    log_vars = {'@loss': None, 'lr': None}

    for batch_idx, batch in enumerate(train_data_loader, start=1):
        total_batches = (epoch - 1) * nb + batch_idx

        x, indiv_mels, mel, y = batch

        x = x.to(args.local_rank, non_blocking=True)
        indiv_mels = indiv_mels.to(args.local_rank, non_blocking=True)
        mel = mel.to(args.local_rank, non_blocking=True)
        y = y.to(args.local_rank, non_blocking=True)

        optimizer.zero_grad()

        pred_y = model(indiv_mels, x)

        sync_weight = criterion['sync_loss'].loss_weight
        sync_loss = criterion['sync_loss'](mel, pred_y) if sync_weight != 0 else 0

        recon_loss = criterion['recon_loss'](pred_y, y) if 'recon_loss' in criterion.keys() else 0

        perceptual_loss = criterion['perceptual_loss'](pred_y, y) if 'perceptual_loss' in criterion.keys() else 0

        loss = sync_loss * sync_weight + (recon_loss + perceptual_loss) * (1 - sync_weight)

        loss.backward()
        log_vars['sync_loss'] = sync_loss
        log_vars['recon_loss'] = recon_loss
        log_vars['perceptual_loss'] = perceptual_loss
        log_vars['lr'] = scheduler.get_last_lr()[0]
        log_vars['@loss'] = loss

        optimizer.step()
        scheduler.step()

        time_meter.update()
        losses_meter.update(log_vars, x.size(0))

        if batch_idx % args.log_steps == 0:
            time_meter.complete_time(nb - batch_idx)
            tb_writer(writer=writer, loss_dict=log_vars, nb=total_batches, tag='train')
            s = f"Epoch:{epoch:{' '}{'>'}{2}d}/{args.epochs} " \
                f"iter:{batch_idx:{' '}{'>'}{len(str(nb))}d}/{nb:d}({batch_idx / nb:.02%}) " \
                f"est. {time_meter.remain_time} {loss_printer(log_vars, fmt='.04e')}"
            logger.info(s)
    logger.info(f"Epoch{epoch:{' '}{'>'}{2}d}/{args.epochs} finished. SyncLoss: {losses_meter.avg}")


def main(args):
    logger = get_logger(args.job_dir)
    device = args.local_rank

    # init wandb
    if args.rank == 0:
        # wandb.tensorboard.patch(root_logdir=args.job_dir)
        wandb.init(project='lip-sync', dir=args.job_dir, name=args.job_dir.split('/')[-1], config=vars(args))

    # Create job and tb_writer
    writer = SummaryWriter(args.job_dir) if args.rank == 0 else None

    # Load dataset
    logger.info(f"Load Dataset")
    train_data_loader, train_sampler, eval_data_loaders, eval_samplers = create_dataloader(args)

    # Create generator
    logger.info(f"Create Model")
    model = Wav2Lip()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model {model} :[Trainable Parameters: {trainable_params}]")

    # Loss function
    logger.info(f"Load loss function")
    criterion = create_criterions(args)

    # allocate model to gpu
    if args.distributed:
        logger.info("Distributed Training")
        model = DDP(model.to(device), device_ids=[device], output_device=device)
    else:
        model.to(device)

    # create optimizers and schedulers
    [optimizer], [scheduler] = create_optim_scheduler(model, args=args, num_batches=len(train_data_loader))

    # Load ckpt
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location='cpu')
        ckpt_loader(ckpt, model=model, optimizer=optimizer, scheduler=scheduler)
        start_epoch = ckpt['epoch'] - 1
        logger.info(f'Load checkpoint from {args.ckpt}. Resume from epoch {start_epoch}')
    else:
        start_epoch = 0

    # Load state_dict
    if args.weight:
        ckpt = torch.load(args.weight, map_location='cpu')
        if args.distributed:
            model.module.load_state_dict(ckpt)
        else:
            model.load_state_dict(ckpt)
        logger.info(f"Load weight from {args.weight}")

    logger.info(attr_extractor(args))

    if args.weight:
        evaluate = model.evaluate if hasattr(model, 'evaluate') else model.module.evaluate
        sync_avg = evaluate(model=model, eval_data_loaders=eval_data_loaders,
                            epoch=0, criterions=criterion,
                            writer=writer, args=args, logger=logger)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        # Train
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train(model, optimizer, scheduler, criterion, train_data_loader, epoch, writer, args, logger)
        # Eval model
        evaluate = model.evaluate if hasattr(model, 'evaluate') else model.module.evaluate
        sync_avg = evaluate(model=model, eval_data_loaders=eval_data_loaders,
                            epoch=epoch, criterions=criterion,
                            writer=writer, args=args, logger=logger)
        if sync_avg < 0.75:
            criterion['sync_loss'].loss_weight = 0.03

        # save model weight
        state_dict_saver(os.path.join(args.job_dir, 'weights', f'{args.model}.pt'), model)
        ckpt_saver(os.path.join(args.job_dir, "ckpt", f"{args.model}_latest.pth"),
                   model=model,
                   optimizer=optimizer, scheduler=scheduler,
                   epoch=epoch)

    logger.info(f"Finish Training")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arguments_parser(parser)

    # Parse arguments
    args = parser.parse_args()
    init_process(args)

    # read from config file
    config.update_params(args)

    # create logger
    logger = get_logger(file_path=args.job_dir)
    args.logger = logger

    main(args)
