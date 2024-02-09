import argparse
import torch


from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from common.meters import AverageMeter
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
    :param args: params
    :param logger: logger
    """

    model.eval()
    for eval_data_name, eval_data_loader in eval_data_loaders:
        loss_dict = test(eval_data_loader, model, args)
        log_string = eval_tb_writer(writer=writer, loss_dict=loss_dict, nb=epoch, eval_data_name=eval_data_name)
        logger.info(log_string)
    logger.info(f"Finish Epoch {epoch} Evaluation\n")


@torch.no_grad()
def test(dataloader: DataLoader,
         model: torch.nn.Module,
         args: argparse.Namespace):
    loss_dict = dict()
    loss_dict["mage_loss"] = AverageMeter()
    loss_dict["acc"] = AverageMeter()

    for idx, (x, y) in enumerate(dataloader, start=1):
        bsz = x.size(0)
        x = x.to(args.local_rank, non_blocking=True)
        y = y.to(args.local_rank, non_blocking=True)

        (loss, acc), imgs, token_all_mask = model(x)

        mage_loss = reduce_all(loss)
        mage_accuracy = reduce_all(acc)
        loss_dict['mage_loss'].update(mage_loss.item(), bsz)
        loss_dict['acc'].update(mage_accuracy.item(), bsz)
    return loss_dict
