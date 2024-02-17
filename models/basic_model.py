from abc import ABC
from enum import Enum

import torch
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel

from ema_pytorch import EMA

from common.meters import AverageMeter
from utils.logging_tool import get_logger
from utils import ckpt_loader, get_dist_info


class CompileMode(Enum):
    DEFAULT = 'default'
    REDUCE_OVERHEAD = 'reduce-overhead'
    MAX_AUTOTUNE = 'max-autotune'


class BasicModel(ABC):

    def __init__(self, *args, **kwargs):
        self.data_timer = AverageMeter()
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
    def compile(*models, mode='default'):
        if torch.cuda.is_available():
            logger = get_logger()
            device_cap = torch.cuda.get_device_capability()
            if device_cap in ((7, 0), (8, 0), (9, 0)):
                for m in models:
                    logger.info(f"Compile {m} in mode {mode}")
                    m = torch.compile(model=m, mode=mode)


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
