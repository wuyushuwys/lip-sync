from abc import ABC
from torch.nn.parallel import DataParallel, DistributedDataParallel
from ema_pytorch import EMA


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

    @staticmethod
    def model_no_ddp(model):
        if isinstance(model, (DistributedDataParallel, DataParallel)):
            return model.module
        return model

    @staticmethod
    def create_ema(model, **kwargs):
        return EMA(model=model, **kwargs)
