import logging
import time
import os

from utils.init_utils import get_dist_info

initialized_logger = {}


def get_logger(file_path):
    rank, _ = get_dist_info()
    name = f"{file_path}:{rank}"
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger

    for logger_name in logger_initialized:
        if file_path.startswith(logger_name):
            return logger

    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    format_str = f'%(asctime)s::%(message)s'
    formatter = logging.Formatter(format_str, "%Y-%m-%d %H:%M:%S")
    os.makedirs(file_path, exist_ok=True)

    if rank == 0:
        file_handler = logging.FileHandler(f"{file_path}/result.log", 'w')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO)
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True
    return logger
