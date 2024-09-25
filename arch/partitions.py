import numpy as np

from einops import rearrange
from utils.logging_tool import get_logger


class GroupPartition:
    def __init__(self, partitions):
        logger = get_logger()
        self.partitions = partitions
        self.shape = None
        logger.info(f"Partition: {partitions}")

    def partition(self, x):  # B HW C
        self.shape = x.shape
        # x = rearrange(x, 'b l (p n) -> b (l p) n', p=self.partitions)
        # x = x.reshape(-1, x.shape[-1] // self.partitions)
        x = x[..., None, :]
        x = x.reshape(*self.shape[:-1], self.partitions, self.shape[-1] // self.partitions)
        x = x.flatten(0, -2)
        return x

    def unpartition(self, x, shape=None):  # L C
        if shape is None:
            # x = rearrange(x, '(b l) n -> b l n', b=b)
            # x = rearrange(x, 'b (l p) n -> b l (p n)', p=self.partitions)
            # x = x.reshape(self.shape[:-1], self.partitions, self.shape[-1] // self.partitions)
            x = x.reshape(self.shape)
        else:
            # print(x.shape)
            # b = shape[0]
            # x = rearrange(x, '(b l) n -> b l n', b=b)
            # print(x.shape)
            # x = rearrange(x, 'b (l p) n -> b l (p n)', p=self.partitions)
            x = x.reshape(shape)
        return x


class LayerPartition:
    def __init__(self, partitions):
        super(LayerPartition, self).__init__()
        self.partitions = partitions
        self.shape = None

    def partition(self, x):  # B HW C
        x = x.permute(0, 3, 1, 2).contiguous()
        self.shape = x.shape
        x = x.reshape(-1, x.shape[-1] * x.shape[-2] // self.partitions)
        return x

    def unpartition(self, x):  # L HW
        x = x.reshape(self.shape)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x


class CustomPartition:
    def __init__(self, partitions):
        super(CustomPartition, self).__init__()
        self.partitions = partitions
        self.shape = None

    def partition(self, x):
        assert x.shape[-1] % 2 == 0
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3] // 2, 2)
        x = x.permute(0, 1, 2, 4, 3).contiguous()
        self.shape = x.shape
        x = x.reshape(-1, x.shape[-1] * 2 // self.partitions)
        return x

    def unpartition(self, x):
        x = x.reshape(self.shape)
        x = x.permute(0, 1, 2, 4, 3).contiguous()
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], -1)
        return x
