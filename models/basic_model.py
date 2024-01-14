from abc import ABC


class BasicModel(ABC):

    def __init__(self, *args, **kwargs):
        pass

    def init_trainer(self, *args, **kwargs):
        pass

    def training_epoch(self, epoch):
        pass

    def evaluating_epoch(self, epoch):
        pass
