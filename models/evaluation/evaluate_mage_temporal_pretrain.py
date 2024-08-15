import argparse
import os
from pathlib import Path

import torch

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from torchvision.utils import save_image, make_grid

import wandb

from common.meters import AverageMeter
from utils.helpers import reduce_all
from utils.logger_utils import eval_tb_writer
from utils.init_utils import master_only
from logging import Logger
from einops import rearrange


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
        loss_dict = test(eval_data_loader, model, epoch, img_folder, args)
        log_string = eval_tb_writer(writer=writer, loss_dict=loss_dict, nb=epoch, eval_data_name=eval_data_name)
        logger.info(log_string)
    logger.info(f"Finish Epoch {epoch} Evaluation\n")


@torch.no_grad()
def test(dataloader: DataLoader,
         model: torch.nn.Module,
         epoch: int,
         img_folder: str,
         args: argparse.Namespace):
    loss_dict = dict()
    loss_dict["mage_loss"] = AverageMeter()
    loss_dict["acc"] = AverageMeter()

    for idx, (x, _, _, y) in enumerate(dataloader, start=1):
        bsz = x.size(0)
        x, _ = map(lambda data: rearrange(data, 'b c t h w -> (b t) c h w'),
                   torch.split(x, 3, dim=1))
        y = rearrange(y, 'b c t h w -> (b t) c h w')
        x = x.to(args.local_rank, non_blocking=True)
        y = y.to(args.local_rank, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            (loss, acc), imgs, token_all_mask = model(x, generate=True, num_batch=bsz)

        mage_loss = reduce_all(loss)
        mage_accuracy = reduce_all(acc)
        loss_dict['mage_loss'].update(mage_loss.item(), bsz)
        loss_dict['acc'].update(mage_accuracy.item(), bsz)

        # scale image from [-1, 1] to [0, 1] for saving
        imgs = (imgs + 1) / 2
        y = (y + 1) / 2

        save_sample_fig(imgs, y, idx, img_folder, num_batch=bsz)

        return loss_dict


@master_only
def save_sample_fig(g, gt, batch_num, folder_path, num_batch):
    outputs = torch.cat([g, gt], dim=-1)
    outputs = rearrange(outputs, '(b f) c h w -> f b c h w', b=num_batch)
    if not os.path.exists(folder_path): os.makedirs(folder_path, exist_ok=True)
    frames = []
    for output in outputs:
        output = make_grid(output, nrow=4, padding=10).permute(1, 2, 0).clamp(0, 1)
        output = (output.cpu() * 255).numpy().astype(np.uint8)
        frames.append(output)
    export_video(frames, output_file=f"{folder_path}/{batch_num}.mp4", fps=25)
    if batch_num == 1 and wandb.run is not None:
        video = wandb.Video(f"{folder_path}/{batch_num}.mp4", fps=25)
        wandb.log({f"{Path(folder_path).stem}": video}, commit=False)
