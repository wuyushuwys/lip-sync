import argparse
import os

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
from omegaconf import OmegaConf

import config

from utils.args_parser import arguments_parser
from utils.init_utils import init_process
from utils.train_utils import create_dataloader, create_criteria, create_optim_scheduler
from utils.logger_utils import attr_extractor
from utils.logging_tool import get_logger

from arch.wav2lip_arch import Wav2Lip
from arch.discriminator_arch import UNetDiscriminatorSN
from models.lipsync_gan_model import LipSyncGAN


def main(args):
    logger = get_logger()
    device = args.local_rank

    # init wandb
    if args.rank == 0 and args.get("use_wandb", True):
        wandb.init(project='lip-sync', dir=args.job_dir, name=args.job_dir.split('/')[-1],
                   config=OmegaConf.to_container(args))
    # Create job and tb_writer
    writer = SummaryWriter(args.job_dir) if args.rank == 0 else None

    # Load dataset
    logger.info(f"Load Dataset")
    train_data_loader, train_sampler, eval_data_loaders, eval_samplers = create_dataloader(args)

    # Create generator
    logger.info(f"Create Model")
    g_model = Wav2Lip()
    d_model = UNetDiscriminatorSN()

    trainable_params = sum(p.numel() for p in g_model.parameters() if p.requires_grad)
    logger.info(f"G-Model {g_model} :[Trainable Parameters: {trainable_params}]")
    trainable_params = sum(p.numel() for p in d_model.parameters() if p.requires_grad)
    logger.info(f"D-Model {d_model} :[Trainable Parameters: {trainable_params}]")

    # Loss function
    logger.info(f"Load loss function")
    criteria = create_criteria(args)

    # allocate model to gpu
    if args.distributed:
        logger.info("Distributed Training")
        g_model = DDP(g_model.to(device), device_ids=[device], output_device=device)
        d_model = DDP(d_model.to(device), device_ids=[device], output_device=device)
    else:
        g_model.to(device)
        d_model.to(device)

    # create optimizers and schedulers
    [g_optimizer, d_optimizer], [g_scheduler, d_scheduler] = create_optim_scheduler(g_model, d_model,
                                                                                    args=args,
                                                                                    num_batches=len(train_data_loader))

    trainer = LipSyncGAN(g_model=g_model,
                         g_optimizer=g_optimizer,
                         g_scheduler=g_scheduler,
                         d_model=d_model,
                         d_optimizer=d_optimizer,
                         d_scheduler=d_scheduler,
                         criteria=criteria,
                         train_data_loader=train_data_loader,
                         eval_data_loaders=eval_data_loaders,
                         logger=logger,
                         args=args,
                         writer=writer)

    # Load ckpt
    start_epoch = trainer.load_ckpt(args.ckpt,
                                    g_model=g_model, g_optimizer=g_optimizer, g_scheduler=g_scheduler,
                                    d_model=d_model, d_optimizer=d_optimizer, d_scheduler=d_scheduler)

    # Load state_dict
    trainer.load_model(model=g_model, ckpt_path=args.g_weight)
    trainer.load_model(model=d_model, ckpt_path=args.d_weight)

    # optimize model graph
    trainer.compile_model()

    logger.info(attr_extractor(args))

    if args.weight or args.ckpt:
        trainer.evaluating_epoch(epoch=start_epoch)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        # Train
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        trainer.training_epoch(epoch=epoch)
        # Eval model
        trainer.evaluating_epoch(epoch=epoch)

        # save model weight
        trainer.save_model(os.path.join(args.job_dir, 'weights'))
        trainer.save_ckpt(os.path.join(args.job_dir, "ckpt"), epoch=epoch)

    logger.info(f"Finish Training")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arguments_parser(parser)

    # Parse arguments
    args = parser.parse_args()
    init_process(args)

    # read from config file
    args = config.update_params(args)

    # create logger
    logger = get_logger(file_path=args.job_dir)

    main(args)
