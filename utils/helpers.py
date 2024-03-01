import functools
import torch.distributed as dist
from inspect import isfunction
from einops import rearrange


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def reduce_all(x):
    if dist.is_initialized():
        world_size = dist.get_world_size()
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x = x / world_size

    return x


def compute_per_image(func):
    @functools.wraps(func)
    def wrapper(x, y):
        assert x.shape == y.shape, f"{x.shape}, {y.shape}"
        if x.dim() == 4:
            assert x.size(1) == 3 or x.size(1) == 1, f"Image Channel Error {x.shape}"
        elif x.dim() == 5:
            x, y = map(lambda t: rearrange(t, 'b c t h w -> (b t) c h w'), (x, y))
        else:
            NotImplementedError(f'Got unexpected input shape {x.shape}')
        return func(x, y)

    return wrapper
