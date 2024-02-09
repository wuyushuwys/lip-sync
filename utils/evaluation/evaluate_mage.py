import argparse
import os
import torch
import wandb

from typing import Dict
from functools import partial
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from pytorch_msssim import SSIM, MS_SSIM

from common.meters import AverageMeter
from utils.helpers import compute_per_image
from utils.init_utils import master_only
from utils.logger_utils import eval_tb_writer
from logging import Logger
from losses import LPIPSLoss

from .utils import reduce_all
from .metrics import calculate_psnr_pt

psnr = compute_per_image(partial(calculate_psnr_pt, crop_border=4))
psnr_y = compute_per_image(partial(calculate_psnr_pt, crop_border=4, test_y_channel=True))


@torch.no_grad()
def evaluation(model, eval_data_loaders, epoch, criterions,
               writer: SummaryWriter, args: argparse.Namespace, logger: Logger):
    """
    Evaluate Generator in eval datasets
    :param model: Generator model
    :param eval_data_loaders: list of eval dataloader
    :param epoch: current epoch
    :param criterions: criterions for evaluation
    :param writer: tb_writer
    :param args: params
    :param logger: logger
    """

    model.eval()
    for eval_data_name, eval_data_loader in eval_data_loaders:
        img_folder = f"{args.job_dir}/samples/{eval_data_name}"
        loss_dict = test(eval_data_loader, model, args)
        log_string = eval_tb_writer(writer=writer, loss_dict=loss_dict, nb=epoch, eval_data_name=eval_data_name)
        logger.info(log_string)
    logger.info(f"Finish Epoch {epoch} Evaluation\n")


@torch.no_grad()
def test(dataloader: DataLoader,
         model: torch.nn.Module,
         args: argparse.Namespace):
    loss_dict = dict()
    loss_dict["loss"] = AverageMeter()

    for idx, (x, y) in enumerate(dataloader, start=1):
        bsz = x.size(0)
        x = x.to(args.local_rank, non_blocking=True)
        y = y.to(args.local_rank, non_blocking=True)

        loss, imgs, token_all_mask = model(x)

        mage_loss = reduce_all(loss)
        loss_dict['mage_loss'].update(mage_loss.item(), bsz)

    return loss_dict
