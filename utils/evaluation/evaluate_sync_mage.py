import argparse
from typing import Dict
from functools import reduce

import torch

from einops import rearrange
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from common.meters import AverageMeter
from utils.logger_utils import eval_tb_writer
from logging import Logger

from .utils import reduce_all


@torch.no_grad()
def evaluation(model, eval_data_loaders, epoch, criteria,
               writer: SummaryWriter, args: argparse.Namespace, logger: Logger, mask):
    model.eval()
    sync_losses = {}
    for eval_data_name, eval_data_loader in eval_data_loaders:
        loss_dict = test(eval_data_loader, model, criteria, args, mask)
        log_string = eval_tb_writer(writer=writer, loss_dict=loss_dict, nb=epoch, eval_data_name=eval_data_name)
        logger.info(log_string)
        sync_losses['eval_data_name'] = loss_dict['sync_loss']
    logger.info(f"Finish Epoch {epoch} Evaluation\n")
    return reduce(lambda x, y: x + y, sync_losses.values()).avg


@torch.no_grad()
def test(dataloader: DataLoader,
         model: torch.nn.Module,
         criteria: Dict,
         args: argparse.Namespace,
         mask):
    loss_dict = {"sync_loss": AverageMeter()}

    for x, indiv_mels, mel, y in dataloader:

        x = x.to(args.local_rank, non_blocking=True)

        # REF is the frame from the same video clip with X but not identical
        x, ref = map(lambda data: rearrange(data, 'b c t h w -> (b t) c h w'),
                     torch.split(x, 3, dim=1))

        assert x.dim() == 4 and x.size(1) == 3, f"Expected get BCHW input shape, but got {x.shape}"

        indiv_mels = indiv_mels.to(args.local_rank, non_blocking=True)
        audio_mel = rearrange(indiv_mels, 'b t c h w -> (b t) c h w')

        y = y.to(args.local_rank, non_blocking=True)

        y = rearrange(y, 'b t -> (b t)')
        x_masked = mask(x.clone(), mask_face=False)

        a, v = model(x_masked, gt=x, audio=audio_mel)

        sync_loss = reduce_all(criteria["sync_loss"](a, v, y))
        loss_dict['sync_loss'].update(sync_loss.item(), x.size(0))

    return loss_dict
