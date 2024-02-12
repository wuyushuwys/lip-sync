from abc import ABC

import torch
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel

from ema_pytorch import EMA

from utils.logging_tool import get_logger
from utils import ckpt_loader, get_dist_info


class BasicModel(ABC):

    def __init__(self, *args, **kwargs):
        pass

    def init_trainer(self, *args, **kwargs):
        pass

    def training_epoch(self, epoch):
        pass

    def evaluating_epoch(self, epoch):
        pass

    def save_model(self, path, best=False):
        pass

    def load_model(self, model, ckpt_path):
        if ckpt_path:
            logger = get_logger()
            ckpt = torch.load(ckpt_path, map_location='cpu')
            self.model_no_ddp(model).load_state_dict(ckpt)

            logger.info(f"{self.model_no_ddp(model)} load weight from {ckpt_path}")

    @staticmethod
    def load_ckpt(ckpt_path, **kwargs):
        """
        Load ckpt if ckpt_path is not None
        Args:
            ckpt_path: ckpt path
            **kwargs: model and its name

        Returns:

        """
        # Load ckpt
        if ckpt_path:
            logger = get_logger()
            ckpt = torch.load(ckpt_path, map_location='cpu')
            ckpt_loader(ckpt, **kwargs)
            start_epoch = ckpt['epoch'] - 1
            logger.info(f'Load checkpoint from {ckpt_path}. Resume from epoch {start_epoch}')
        else:
            start_epoch = 0
        return start_epoch

    @staticmethod
    def model_no_ddp(model):
        if isinstance(model, (DistributedDataParallel, DataParallel)):
            return model.module
        return model

    @staticmethod
    def create_ema(model, **kwargs):
        return EMA(model=model, **kwargs)

    @staticmethod
    @torch.no_grad()
    def reduce_loss_dict(loss_dict):
        """reduce loss dict.

        In distributed training, it averages the losses among different GPUs .

        Args:
            loss_dict (OrderedDict): Loss dict.
        """
        rank, world_size = get_dist_info()
        if world_size > 1:
            keys = []
            losses = []
            for name, value in loss_dict.items():
                if torch.is_tensor(value):
                    keys.append(name)
                    losses.append(value)
            losses = torch.stack(losses, 0)
            dist.all_reduce(losses)
            losses /= world_size
            for key, loss in zip(keys, losses):
                loss_dict[key] = loss

        return loss_dict
