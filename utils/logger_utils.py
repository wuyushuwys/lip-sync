import importlib
import torch
import wandb

from torch.utils import tensorboard
from thop import profile, clever_format

import models

from utils.init_utils import master_only, when_attr_is_true

__all__ = ["attr_extractor", "loss_printer", "tb_writer", "profile_model"]


@master_only
def attr_extractor(obj):
    from utils.logging_tool import LoggingTool
    attrs = list(filter(lambda x: not x.startswith('_'), dir(obj)))  # Remove default and help attributes
    attr_dict = dict()
    info_len = 40
    total_len = 120
    string = f"\n{'INFO':{'*'}{'^'}{total_len}s}\n"
    str_head = '** '

    def attrs2dict(attr):
        for k, v in attr.items():
            if isinstance(v, dict) and 'type' in v:
                k = f"{k}[{v.pop('type')}]"
            attr_dict[k] = v

    for name in attrs:
        attr = getattr(obj, name)
        if name == "losses":
            # if isinstance(attr, dict):
            attrs2dict(attr)
        else:
            if isinstance(attr, dict) and 'type' in attr:
                name = f"{name}[{attr.pop('type')}]"
            attr_dict[name] = attr

    for k, v in attr_dict.items():
        if isinstance(v, LoggingTool):
            v = v.name
        if inspect.isfunction(v):
            v = inspect.getsource(v)
        v_str = str(v)
        string += f"{str_head}{f'{k}:':{''}{'<'}{info_len}s}{v_str}\n"

    string += f"{'':{'*'}{'^'}{total_len}s}\n"
    return string


@master_only
def loss_printer(loss_dict: dict, fmt='.4f'):
    s = ''
    for k, v in loss_dict.items():
        if k != 'loss':
            s += f"{k}:{v.item():{fmt}}  " if hasattr(v, 'item') else f"{k}:{v:{fmt}}  "

    return f"[{s.rstrip()}]"


@master_only
def tb_writer(writer: tensorboard.writer, loss_dict: dict, nb: int, tag: str = 'train'):
    for k, v in loss_dict.items():
        if torch.is_tensor(v):
            v = v.item()
        writer.add_scalar(f'{tag}/{k}', v, nb)
        wandb.log({f'{tag}/{k}': v})


@when_attr_is_true('profile')
@master_only
def profile_model(args):
    model = None
    input = None
    macs, param = clever_format(profile(model, inputs=(input,), verbose=False))
    args.logger.info(f"Model :[ #MACs: {macs}\t #Params: {param}]")
