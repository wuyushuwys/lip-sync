import argparse
import os
import torch
from typing import Dict
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from common.meters import AverageMeter
from utils.init_utils import master_only
from utils.logger_utils import eval_tb_writer
from logging import Logger

from .utils import reduce_all


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
        sync_losses[eval_data_name] = loss_dict['sync_loss'].avg
    logger.info(f"Finish Epoch {epoch} Evaluation\n")
    return torch.tensor(list(sync_losses.values())).mean()


@torch.no_grad()
def test(dataloader: DataLoader,
         model: torch.nn.Module,
         criterions: Dict,
         epoch: int,
         img_folder: str,
         args: argparse.Namespace):
    loss_dict = {
        "sync_loss": AverageMeter(),
        "recon_loss": AverageMeter(),
        "perceptual_loss": AverageMeter(),
    }

    for idx, (x, indiv_mels, mel, y) in enumerate(dataloader, start=1):
        bsz = x.size(0)
        x = x.to(args.local_rank, non_blocking=True)
        indiv_mels = indiv_mels.to(args.local_rank, non_blocking=True)
        mel = mel.to(args.local_rank, non_blocking=True)
        y = y.to(args.local_rank, non_blocking=True)

        pred_y = model(indiv_mels, x)
        sync_loss = reduce_all(criterions['sync_loss'](mel, pred_y))
        recon_loss = reduce_all(criterions['recon_loss'](pred_y, y))
        perceptual_loss = reduce_all(criterions['perceptual_loss'](pred_y, y))

        loss_dict['sync_loss'].update(sync_loss, bsz)
        loss_dict['recon_loss'].update(recon_loss, bsz)
        loss_dict['perceptual_loss'].update(perceptual_loss, bsz)
        save_sample_images(x, pred_y, y, idx, epoch=epoch, folder_path=img_folder)

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
