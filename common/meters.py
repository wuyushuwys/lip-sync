"""Meters."""
import time
import datetime

import torch

from utils import loss_printer


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, sum=0, count=0):
        self._val = 0
        self._avg = 0
        self._sum = sum
        self._count = count

    def reset(self):
        self._val = 0
        self._avg = 0
        self._sum = 0
        self._count = 0

    def update(self, val, n=1):
        self._val = val
        self._sum += val * n
        self._count += n

    @property
    def avg(self):
        if self._count != 0:
            return self._sum / self._count
        else:
            return 0

    @property
    def val(self):
        return self._val

    @property
    def sum(self):
        return self._sum

    @property
    def count(self):
        return self._count

    def __add__(self, other):
        return AverageMeter(sum=self._sum + other.sum, count=self.count + other.count)


class TimeMeter:
    """Computes and stores the average and current value"""

    def __init__(self, total_iterations=None):
        self.start_time = time.monotonic()
        self.end_time = self.start_time
        self.sum = 0
        self.avg = 0
        self.count = 0
        self.remain_time = 0
        self.total_iterations = total_iterations

    def reset(self):
        self.start_time = time.monotonic()
        self.end_time = self.start_time
        self.sum = 0
        self.avg = 0
        self.count = 0

    def update(self, n=1):
        self.end_time = time.monotonic()
        self.sum = self.end_time - self.start_time
        self.count += n
        self.avg = self.sum / self.count

    def update_count(self, count):
        self.end_time = time.monotonic()
        self.sum = self.end_time - self.start_time
        self.count += count
        self.avg = self.sum / self.count

    def tic(self):
        self.start_time = time.monotonic()

    def toc(self):
        self.end_time = time.monotonic()

    def eta(self):
        remain_batch = self.total_iterations - self.count
        self.remain_time = datetime.timedelta(seconds=int(self.avg * remain_batch))
        return self.remain_time


class LossesMeter:
    """Computes and stores the average and current value"""

    def __init__(self, fmt='.04f'):
        self._loss = {}
        self.fmt = fmt

    def update(self, loss_dict: dict, size=1):
        for key, value in loss_dict.items():
            value = value.item() if torch.is_tensor(value) else value
            if key not in self._loss.keys():
                self._loss[key] = AverageMeter()
                self._loss[key].update(value, n=size)
            else:
                self._loss[key].update(value, n=size)

    @property
    def avg(self):
        return loss_printer({key: meter.avg for key, meter in self._loss.items()}, fmt=self.fmt)

    @property
    def val(self):
        return loss_printer({key: meter.val for key, meter in self._loss.items()}, fmt=self.fmt)
