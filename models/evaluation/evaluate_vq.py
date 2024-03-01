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
from utils.helpers import compute_per_image, reduce_all
from utils.init_utils import master_only
from utils.logger_utils import eval_tb_writer
from logging import Logger
from losses import LPIPSLoss

from .metrics import calculate_psnr_pt

psnr = compute_per_image(partial(calculate_psnr_pt, crop_border=4))
psnr_y = compute_per_image(partial(calculate_psnr_pt, crop_border=4, test_y_channel=True))


@torch.no_grad()
def evaluation(model, eval_data_loaders, epoch, criteria,
               writer: SummaryWriter, args: argparse.Namespace, logger: Logger):
    """
    Evaluate Generator in eval datasets
    :param model: Generator model
    :param eval_data_loaders: list of eval dataloader
    :param epoch: current epoch
    :param criteria: criteria for evaluation
    :param writer: tb_writer
    :param args: params
    :param logger: logger
    """

    model.eval()
    for eval_data_name, eval_data_loader in eval_data_loaders:
        img_folder = f"{args.job_dir}/samples/{eval_data_name}"
        loss_dict = test(eval_data_loader, model, criteria, epoch, img_folder, args)
        log_string = eval_tb_writer(writer=writer, loss_dict=loss_dict, nb=epoch, eval_data_name=eval_data_name)
        logger.info(log_string)
    logger.info(f"Finish Epoch {epoch} Evaluation\n")


@torch.no_grad()
def test(dataloader: DataLoader,
         model: torch.nn.Module,
         criteria: Dict,
         epoch: int,
         img_folder: str,
         args: argparse.Namespace):
    loss_dict = {k: AverageMeter() for k in criteria.keys() if k in ['perceptual_loss', 'recon_loss']}
    loss_dict["SSIM"] = AverageMeter()
    loss_dict["MS_SSIM"] = AverageMeter()
    loss_dict["PSNR"] = AverageMeter()
    loss_dict["lpips[alex]"] = AverageMeter()
    loss_dict['PSNR_y'] = AverageMeter()
    ssim = compute_per_image(SSIM(data_range=1))
    ms_ssim = compute_per_image(MS_SSIM(data_range=1))
    lpips = LPIPSLoss(loss_weight=1, lpips_loss_arch='alex').to(args.local_rank)

    for idx, (x, y) in enumerate(dataloader, start=1):
        bsz = x.size(0)
        x = x.to(args.local_rank, non_blocking=True)
        y = y.to(args.local_rank, non_blocking=True)

        pred_y, vq_info = model(x)

        pred_y = (pred_y + 1) / 2
        y = (y + 1) / 2

        recon_loss = reduce_all(criteria['recon_loss'](pred_y, y, val=True))
        loss_dict['recon_loss'].update(recon_loss.item(), bsz)

        if 'perceptual_loss' in criteria.keys():
            perceptual_loss = reduce_all(criteria['perceptual_loss'](pred_y, y, normalize=True, val=True))
            loss_dict['perceptual_loss'].update(perceptual_loss.item(), bsz)

        save_sample_images(pred_y, y, idx, epoch, img_folder)

        loss_dict['MS_SSIM'].update(reduce_all(ms_ssim(pred_y, y)), bsz)
        loss_dict['PSNR'].update(reduce_all(psnr(pred_y, y).mean()), bsz)
        loss_dict['PSNR_y'].update(reduce_all(psnr_y(pred_y, y).mean()), bsz)
        loss_dict['SSIM'].update(reduce_all(ssim(pred_y, y)), bsz)
        loss_dict["lpips[alex]"].update(reduce_all(lpips(pred_y, y, True)), bsz)

    return loss_dict


@master_only
def save_sample_images(g, gt, batch_num, epoch, folder_path):
    outputs = torch.cat([g, gt], dim=-1)
    # folder = os.path.join(folder_path, "samples_step{:03d}".format(epoch))
    if not os.path.exists(folder_path): os.makedirs(folder_path, exist_ok=True)
    outputs = make_grid(outputs, nrow=4, padding=10)
    save_image(outputs, fp=f"{folder_path}/{batch_num}.jpg")
    if batch_num == 1 and wandb.run is not None:
        image = wandb.Image(outputs, file_type='jpg')
        wandb.log({f"{Path(folder_path).stem}": image}, commit=False)
