import argparse
import os
import torch
import wandb

from typing import Dict
from functools import partial, reduce
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

from .utils import reduce_all
from .metrics import calculate_psnr_pt

psnr = compute_per_image(partial(calculate_psnr_pt, crop_border=0))
psnr_y = compute_per_image(partial(calculate_psnr_pt, crop_border=0, test_y_channel=True))


@torch.no_grad()
def evaluation(model, eval_data_loaders, epoch, criterions,
               writer: SummaryWriter, args: argparse.Namespace, logger: Logger, mask=None):
    """
    Evaluate Generator in eval datasets
    :param model: Generator model
    :param eval_data_loaders: list of eval dataloader
    :param epoch: current epoch
    :param criterions: criterions for evaluation
    :param writer: tb_writer
    :param device: index of device
    :param args: params
    :param logger: logger
    """

    model.eval()
    sync_losses = {}
    for eval_data_name, eval_data_loader in eval_data_loaders:
        img_folder = f"{args.job_dir}/samples/{eval_data_name}"
        loss_dict = test(eval_data_loader, model, criterions, epoch, img_folder, args, mask)
        log_string = eval_tb_writer(writer=writer, loss_dict=loss_dict, nb=epoch, eval_data_name=eval_data_name)
        logger.info(log_string)
        sync_losses[eval_data_name] = loss_dict['sync_loss']
    logger.info(f"Finish Epoch {epoch} Evaluation\n")
    return reduce(lambda x, y: x + y, sync_losses.values()).avg


@torch.no_grad()
def test(dataloader: DataLoader,
         model: torch.nn.Module,
         criterions: Dict,
         epoch: int,
         img_folder: str,
         args: argparse.Namespace,
         mask=None):
    loss_dict = {k: AverageMeter() for k in criterions.keys()}
    loss_dict["SSIM"] = AverageMeter()
    loss_dict["MS_SSIM"] = AverageMeter()
    loss_dict["PSNR"] = AverageMeter()
    ssim = compute_per_image(SSIM(data_range=1))
    ms_ssim = compute_per_image(MS_SSIM(data_range=1))

    for idx, (x, indiv_mels, mel, y) in enumerate(dataloader, start=1):
        bsz = x.size(0)
        x = x.to(args.local_rank, non_blocking=True)
        indiv_mels = indiv_mels.to(args.local_rank, non_blocking=True)
        mel = mel.to(args.local_rank, non_blocking=True)
        y = y.to(args.local_rank, non_blocking=True)

        if mask is not None:
            try:
                x = mask(x)
            except RuntimeError as e:
                args.logger.error(f'failed at face masking {e}')
                continue

        pred_y = model(indiv_mels, x)

        sync_loss = reduce_all(criterions['sync_loss'](mel, pred_y))
        loss_dict['sync_loss'].update(sync_loss.item(), bsz)

        recon_loss = reduce_all(criterions['recon_loss'](pred_y, y, val=True))
        loss_dict['recon_loss'].update(recon_loss.item(), bsz)

        if 'perceptual_loss' in criterions.keys():
            perceptual_loss = reduce_all(criterions['perceptual_loss'](pred_y, y, val=True))
            loss_dict['perceptual_loss'].update(perceptual_loss.item(), bsz)

        loss_dict['MS_SSIM'].update(reduce_all(ms_ssim(pred_y, y)), bsz)
        save_sample_images(x, pred_y, y, idx, epoch=epoch, folder_path=img_folder)

        loss_dict['PSNR'].update(reduce_all(psnr(pred_y, y).mean()), bsz)
        loss_dict['SSIM'].update(reduce_all(ssim(pred_y, y)), bsz)

    return loss_dict


@master_only
def save_sample_images(x, g, gt, batch_num, epoch, folder_path):
    refs, inps = torch.split(x, 3, dim=1)
    outputs = torch.cat([refs, inps, g, gt], dim=-1).unbind(2)
    outputs = torch.cat(outputs, dim=-2)

    # folder = os.path.join(folder_path, "samples_step{:03d}".format(epoch))
    if not os.path.exists(folder_path): os.makedirs(folder_path, exist_ok=True)
    outputs = make_grid(outputs, nrow=4, padding=10)
    save_image(outputs, fp=f"{folder_path}/{batch_num}.jpg")
    if batch_num == 1:
        image = wandb.Image(outputs, file_type='jpg')
        wandb.log({f"{Path(folder_path).stem}": image})
