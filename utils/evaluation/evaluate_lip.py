import argparse
import os
import torch

from typing import Dict
from functools import partial, reduce
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import save_image
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
               writer: SummaryWriter, args: argparse.Namespace, logger: Logger):
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
        loss_dict = test(eval_data_loader, model, criterions, epoch, img_folder, args)
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
         args: argparse.Namespace):
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

        pred_y = model(indiv_mels, x)

        sync_loss = reduce_all(criterions['sync_loss'](mel, pred_y))
        loss_dict['sync_loss'].update(sync_loss.item(), bsz)

        recon_loss = reduce_all(criterions['recon_loss'](pred_y, y, eval=True))
        loss_dict['recon_loss'].update(recon_loss.item(), bsz)

        if 'perceptual_loss' in criterions.keys():
            perceptual_loss = reduce_all(criterions['perceptual_loss'](pred_y, y, eval=True))
            loss_dict['perceptual_loss'].update(perceptual_loss.item(), bsz)

        loss_dict['MS_SSIM'].update(ms_ssim(pred_y, y))
        save_sample_images(x, pred_y, y, idx, epoch=epoch, folder_path=img_folder)

        # b, c, t, h, w = pred_y.size()
        # if 'crop_pad' in args.video_spec.keys():
        #     y1, y2, x1, x2 = args.video_spec['crop_pad']
        #     pred_y = pred_y[..., h // 2 + y1: y2, x1: x2]
        #     y = y[..., h // 2 + y1: y2, x1: x2]
        # else:
        #     pred_y = pred_y[..., h // 2:, :]
        #     y = y[..., h // 2:, :]

        loss_dict['PSNR'].update(psnr(pred_y, y).mean(), bsz)
        loss_dict['SSIM'].update(ssim(pred_y, y))

    return loss_dict


@master_only
def save_sample_images(x, g, gt, batch_num, epoch, folder_path):
    refs, inps = torch.split(x, 3, dim=1)
    outputs = torch.cat([refs, inps, g, gt], dim=-1).unbind(2)
    outputs = torch.cat(outputs, dim=-2)

    folder = os.path.join(folder_path, "samples_step{:03d}".format(epoch))
    if not os.path.exists(folder): os.makedirs(folder, exist_ok=True)
    save_image(outputs[:1, ...], fp=f"{folder}/{batch_num}.jpg", nrow=1, padding=10)

    # x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    # g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    # gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    # refs, inps = x[..., 3:], x[..., :3]
    # for idx, output in enumerate(outputs)
    # collage = np.concatenate((refs, inps, g, gt), axis=-2)
    # for batch_idx, c in enumerate(collage):
    #     for t in range(len(c)):
    #         cv2.imwrite(f'{folder}/{batch_num}_{batch_idx}_{t}.jpg', c[t])
