import torch.distributed as dist


def reduce_all(x):
    if dist.is_initialized():
        world_size = dist.get_world_size()
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x = x / world_size

    return x