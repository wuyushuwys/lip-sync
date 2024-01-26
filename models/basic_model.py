from abc import ABC

import torch
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel

from ema_pytorch import EMA

from utils.logging_tool import get_logger
from utils import ckpt_loader


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
            ckpt_loader(ckpt_path, **kwargs)
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
