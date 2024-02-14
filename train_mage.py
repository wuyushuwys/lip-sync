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
from utils.train_utils import create_dataloader, create_criteria, create_optim_scheduler, subdict
from utils.logger_utils import attr_extractor
from utils.logging_tool import get_logger

from arch import conditioned_mage_arch
from models.mage_model import MageModel


def main(args):
    logger = get_logger()
    device = args.local_rank

    # init wandb
    if args.rank == 0:
        wandb.init(project='lip-sync', dir=args.job_dir, name=args.job_dir.split('/')[-1],
                   config=OmegaConf.to_container(args))
    # Create job and tb_writer
    writer = SummaryWriter(args.job_dir) if args.rank == 0 else None

    # Load dataset
    logger.info(f"Load Dataset")
    train_data_loader, train_sampler, eval_data_loaders, eval_samplers = create_dataloader(args)

    # Create generator
    logger.info(f"Create Model")

    model = conditioned_mage_arch.__dict__[args.model.get("type")](**subdict(args.model, 'type'))
    # model = lip_mage_vit_small(**args.model)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model {model} :[Trainable Parameters: {trainable_params}]")

    # Loss function
    logger.info(f"Load loss function")
    criteria = create_criteria(args)

    # allocate model to gpu
    if args.distributed:
        logger.info("Distributed Training")
        model = DDP(model.to(device), device_ids=[device], output_device=device)
    else:
        model.to(device)

    # Scale learning rate based on effective batch_size (based on mage-pretrain)
    args.optim.lr = args.optim.lr / 256 * args.batch_size * args.world_size
    # create optimizers and schedulers
    [optimizer], [scheduler] = create_optim_scheduler(model,
                                                      args=args,
                                                      num_batches=len(train_data_loader))
    trainer = MageModel(model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        criteria=criteria,
                        train_data_loader=train_data_loader,
                        eval_data_loaders=eval_data_loaders,
                        logger=logger,
                        args=args,
                        writer=writer)

    # Load ckpt
    start_epoch = trainer.load_ckpt(args.ckpt,
                                    model=model, optimizer=optimizer, scheduler=scheduler)

    # Load state_dict
    trainer.load_model(model=model, ckpt_path=args.weight)

    logger.info(attr_extractor(args))

    if args.weight or args.ckpt:
        trainer.evaluating_epoch(epoch=start_epoch)
        if args.eval_only:
            return logger.info('Finish evaluation')

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
