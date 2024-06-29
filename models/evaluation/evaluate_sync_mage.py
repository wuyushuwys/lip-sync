import argparse
import math
from typing import Dict
from functools import reduce

import torch

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from common.meters import AverageMeter
from utils.helpers import reduce_all
from utils.logger_utils import eval_tb_writer
from logging import Logger


@torch.no_grad()
def evaluation(model, eval_data_loaders, epoch, criteria,
               writer: SummaryWriter, args: argparse.Namespace, logger: Logger):
    """
    Evaluate Generator in eval datasets
    :param model: Generator model
    :param eval_data_loaders: list of eval dataloader
    :param epoch: current epoch
    :param criteria: criteria
    :param writer: tb_writer
    :param device: index of device
    :param args: params
    :param logger: logger
    """

    model.eval()
    sync_losses = {}
    for eval_data_name, eval_data_loader in eval_data_loaders:
        bottom_half = args.video_spec.get('bottom_half', False)
        loss_dict = test(eval_data_loader, model, criteria, args)
        log_string = eval_tb_writer(writer=writer, loss_dict=loss_dict, nb=epoch, eval_data_name=eval_data_name)
        logger.info(log_string)
        sync_losses['eval_data_name'] = loss_dict['sync_loss']
    logger.info(f"Finish Epoch {epoch} Evaluation\n")
    return reduce(lambda x, y: x + y, sync_losses.values()).avg


@torch.no_grad()
def test(dataloader: DataLoader,
         model: torch.nn.Module,
         criteria: Dict,
         args: argparse.Namespace):
    loss_dict = {"sync_loss": AverageMeter()}

    for x, mel, y in dataloader:
        x = x.to(args.local_rank, non_blocking=True)
        mel = mel.to(args.local_rank, non_blocking=True)
        y = y.to(args.local_rank, non_blocking=True)

        a, v = model(mel, x)
        sync_loss = reduce_all(criteria["sync_loss"](a, v, y))
        loss_dict['sync_loss'].update(sync_loss.item(), x.size(0))

    return loss_dict
