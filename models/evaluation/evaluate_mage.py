import argparse
import os
from pathlib import Path
from functools import partial, reduce

import torch

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from pytorch_msssim import SSIM, MS_SSIM
from einops import rearrange

import wandb

from common.meters import AverageMeter
from utils.helpers import compute_per_image, reduce_all
from utils.logger_utils import eval_tb_writer
from utils.init_utils import master_only
from logging import Logger

from .metrics import calculate_psnr_pt

psnr = compute_per_image(partial(calculate_psnr_pt, crop_border=0))
psnr_y = compute_per_image(partial(calculate_psnr_pt, crop_border=0, test_y_channel=True))


@torch.no_grad()
def evaluation(model, eval_data_loaders, epoch, criteria,
               writer: SummaryWriter, args: argparse.Namespace, logger: Logger, mask=None):
    """
    Evaluate Generator in eval datasets
    :param model: Generator model
    :param eval_data_loaders: list of eval dataloader
    :param epoch: current epoch
    :param criteria: criteria for evaluation
    :param writer: tb_writer
    :param args: params
    :param logger: logger
    :param mask: mask
    """

    model.eval()
    for eval_data_name, eval_data_loader in eval_data_loaders:
        img_folder = f"{args.job_dir}/samples/{eval_data_name}"
        loss_dict = test(eval_data_loader, model, epoch, img_folder, args, mask)
        log_string = eval_tb_writer(writer=writer, loss_dict=loss_dict, nb=epoch, eval_data_name=eval_data_name)
        logger.info(log_string)
    logger.info(f"Finish Epoch {epoch} Evaluation\n")


@torch.no_grad()
def test(dataloader: DataLoader,
         model: torch.nn.Module,
         epoch: int,
         img_folder: str,
         args: argparse.Namespace,
         mask=None):
    loss_dict = dict()
    loss_dict["mage_loss"] = AverageMeter()
    loss_dict["acc"] = AverageMeter()
    # although mage does not optimize pixel level loss, we still evaluate these metrics
    loss_dict["SSIM"] = AverageMeter()
    loss_dict["MS_SSIM"] = AverageMeter()
    loss_dict["PSNR"] = AverageMeter()

    ssim = compute_per_image(SSIM(data_range=1))
    ms_ssim = compute_per_image(MS_SSIM(data_range=1))

    for idx, (x, indiv_mels, mel, y) in enumerate(dataloader, start=1):
        bsz = x.size(0)
        x = x.to(args.local_rank, non_blocking=True)

        x, ref = map(lambda data: rearrange(data, 'b c t h w -> (b t) c h w'),
                     torch.split(x, 3, dim=1))

        assert x.dim() == 4 and x.size(1) == 3, f"Expected get BCHW input shape, but got {x.shape}"

        indiv_mels = indiv_mels.to(args.local_rank, non_blocking=True)
        audio_mel = rearrange(indiv_mels, 'b t c h w -> (b t) c h w')

        # unused for now
        # mel = mel.to(args.local_rank, non_blocking=True)

        y = y.to(args.local_rank, non_blocking=True)
        y = rearrange(y, 'b c t h w -> (b t) c h w')

        x_masked = mask(x.clone())

        (loss, acc), imgs, token_all_mask = model(x_masked, gt=y, ref=ref, audio=audio_mel, generate=True)

        mage_loss = reduce_all(loss)
        mage_accuracy = reduce_all(acc)
        loss_dict['mage_loss'].update(mage_loss.item(), bsz)
        loss_dict['acc'].update(mage_accuracy.item(), bsz)

        vq_y, _ = model.module.vqgan(y)

        # scale image from [-1, 1] to [0, 1] for saving and image pixel evaluation
        x_masked = (x_masked + 1) / 2
        imgs = (imgs + 1) / 2
        y = (y + 1) / 2
        vq_y = (vq_y + 1) / 2

        loss_dict['MS_SSIM'].update(reduce_all(ms_ssim(imgs, y)), bsz)
        loss_dict['PSNR'].update(reduce_all(psnr(imgs, y).mean()), bsz)
        loss_dict['SSIM'].update(reduce_all(ssim(imgs, y)), bsz)

        save_sample_images(x_masked, imgs, vq_y, y, idx, epoch, img_folder)

    return loss_dict


@master_only
def save_sample_images(x, g, vq_gt, gt, batch_num, epoch, folder_path):
    outputs = torch.cat([x, g, vq_gt, gt], dim=-1)
    # folder = os.path.join(folder_path, "samples_step{:03d}".format(epoch))
    if not os.path.exists(folder_path): os.makedirs(folder_path, exist_ok=True)
    outputs = make_grid(outputs, nrow=2, padding=10)
    save_image(outputs, fp=f"{folder_path}/{batch_num}.jpg")
    if batch_num == 1 and wandb.run is not None:
        image = wandb.Image(outputs, file_type='jpg')
        wandb.log({f"{Path(folder_path).stem}": image}, commit=False)
