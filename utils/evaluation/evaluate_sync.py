import argparse

from collections import OrderedDict
from typing import Dict

import torch
import wandb

from common.meters import AverageMeter
from utils.init_utils import master_only
from utils.logging_tool import get_logger
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


@master_only
def evaluation(model, eval_data_loaders, epoch, criterion,
               writer: SummaryWriter, args: argparse.Namespace):
    """
    Evaluate Generator in eval datasets
    :param model: Generator model
    :param eval_data_loaders: list of eval dataloader
    :param epoch: current epoch
    :param criterion: criterion for evaluation
    :param writer: tb_writer
    :param device: index of device
    :param args: params
    """
    # logger = get_logger(args.job_dir)
    model.eval()
    for eval_data_name, eval_data_loader in eval_data_loaders:
        loss_dict = test(eval_data_loader, model, criterion, args, epoch)
        log_string = f"##\tEval: {eval_data_name}"
        for k, v in loss_dict.items():
            log_string += f"\t{k}: {v.avg:.04e}"
            writer.add_scalar(f"eval_{eval_data_name}/{k}", v.avg, epoch)
            wandb.log({f"eval_{eval_data_name}/{k}": v.avg})
        logger.warning(log_string)
    args.logger.warning(f"Finish Epoch {epoch} Evaluation\n")


@torch.no_grad()
def test(dataloader: DataLoader, model: torch.nn.Module, criterion: Dict, args: argparse.Namespace):
    loss_dict = {"sync_loss" : AverageMeter()}
    loss = criterion['sync_loss']

    for x, mel, y in dataloader:
        x = x.to(args.local_rank, non_blocking=True)
        mel = mel.to(args.local_rank, non_blocking=True)
        y = y.to(args.local_rank, non_blocking=True)

        a, v = model(mel, x)

        loss_dict['sync_loss'].update(loss(a, v, y), x.size(0))

    return loss_dict
