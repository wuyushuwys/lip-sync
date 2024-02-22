import argparse
import os
import torch
import wandb
from typing import Dict
from pathlib import Path
from functools import reduce

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from torch.nn.functional import interpolate

from common.meters import AverageMeter
from utils.logger_utils import eval_tb_writer
from utils.init_utils import master_only
from logging import Logger

from .utils import reduce_all


@torch.no_grad()
def evaluation(model, eval_data_loaders, epoch, criteria,
               writer: SummaryWriter, args: argparse.Namespace, logger: Logger,
               mask, vqgan):
    model.eval()
    # sync_losses = {}
    for eval_data_name, eval_data_loader in eval_data_loaders:
        img_folder = f"{args.job_dir}/samples/{eval_data_name}"
        loss_dict = test(eval_data_loader, model, criteria, epoch, img_folder, args, mask, vqgan)
        log_string = eval_tb_writer(writer=writer, loss_dict=loss_dict, nb=epoch, eval_data_name=eval_data_name)
        logger.info(log_string)
        # sync_losses[eval_data_name] = loss_dict['sync_loss']
    logger.info(f"Finish Epoch {epoch} Evaluation\n")
    # return reduce(lambda x, y: x + y, sync_losses.values()).avg


@torch.no_grad()
def test(dataloader: DataLoader,
         model: torch.nn.Module,
         criteria: Dict,
         epoch: int,
         img_folder: str,
         args: argparse.Namespace,
         mask, vqgan):
    loss_dict = dict(
        hard=AverageMeter(),
        # soft=AverageMeter(),
        acc=AverageMeter(),
        k_acc=AverageMeter(),
        gt_mr=AverageMeter(),
        pred_mr=AverageMeter(),
        diff_mr=AverageMeter(),
    )

    for idx, (x, _) in enumerate(dataloader, start=1):
        bsz = x.size(0)
        x = x.to(args.local_rank, non_blocking=True)

        # generate masked x
        masked_x = mask(x)
        # generate latent information for masked x
        latent_mx = vqgan.encode(masked_x)
        qmx, _, qmx_info = vqgan.quantize(z=latent_mx)
        qmx_indices = qmx_info['min_encoding_indices'].reshape(bsz, -1)

        # generate latent information for unmasked x
        latent_x = vqgan.encode(x)
        qx, _, qx_info = vqgan.quantize(z=latent_x)
        qx_indices = qx_info['min_encoding_indices'].reshape(bsz, -1)

        # soft ground truth
        # gt_soft = torch.nn.functional.cosine_similarity(latent_mx.flatten(2), latent_x.flatten(2), dim=1)
        # gt_soft = torch.nn.functional.cosine_similarity(qx.flatten(2), qmx.flatten(2), dim=1) / 2 + 0.5
        # hard ground truth
        gt_hard = qmx_indices.eq(qx_indices).float()
        key_gt = (1 - gt_hard).nonzero(as_tuple=True)
        # print("similar semantic", gt_hard.sum() / gt_hard.numel())

        pred_y = model(masked_x)
        pred_hard = pred_y.greater(0.5).float()

        gt_mask_ratio = reduce_all(1 - (gt_hard.sum() / gt_hard.numel()))
        pred_mask_ratio = reduce_all(1 - (pred_hard.sum() / pred_hard.numel()))

        loss_dict['gt_mr'].update(gt_mask_ratio, bsz)
        loss_dict['pred_mr'].update(pred_mask_ratio, bsz)
        loss_dict['diff_mr'].update(gt_mask_ratio - pred_mask_ratio, bsz)

        # print("pred similar semantic", pred_hard.sum() / pred_hard.numel())

        accuracy = gt_hard.eq(pred_hard).sum() / gt_hard.numel()
        loss_dict['acc'].update(reduce_all(accuracy), bsz)

        key_accuracy = gt_hard.eq(pred_hard)[key_gt].sum() / gt_hard[key_gt].numel()
        loss_dict['k_acc'].update(reduce_all(key_accuracy), bsz)

        loss_hard = torch.nn.functional.binary_cross_entropy(pred_y, gt_hard)
        loss_dict['hard'].update(reduce_all(loss_hard), bsz)

        gt = interpolate(gt_hard.reshape(bsz, 1, 32, 32), (256, 256), mode='nearest')
        y = interpolate(pred_hard.reshape(bsz, 1, 32, 32), (256, 256), mode='nearest')
        save_sample_images(y, gt, idx, None, img_folder)
        # loss_soft = torch.nn.functional.mse_loss(pred_y, gt_soft)
        # loss_dict['soft'].update(reduce_all(loss_soft), bsz)

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
