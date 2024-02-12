import os
import logging
import sys
import threading

from utils.init_utils import get_dist_info

logger_initialized = {}

_lock = threading.RLock()


def _accquire_lock() -> None:
    """Acquire the module-level lock for serializing access to shared data.

    This should be released with _release_lock().
    """
    if _lock:
        _lock.acquire()


def _release_lock() -> None:
    """Release the module-level lock acquired by calling _accquire_lock()."""
    if _lock:
        _lock.release()


def get_logger(name='logger', file_path=None):
    rank, _ = get_dist_info()
    name = f"{name}:{rank}"
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger

    log_level = logging.INFO if rank == 0 else logging.ERROR

    # for name in logger_initialized:
    #     if file_path.startswith(name):
    #         return logger

    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    format_str = f'[%(asctime)s] %(message)s'
    formatter = logging.Formatter(format_str, "%Y-%m-%d %H:%M:%S")
    if file_path:
        os.makedirs(file_path, exist_ok=True)

        if rank == 0:
            file_handler = logging.FileHandler(f"{file_path}/result.log", 'w')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    if rank == 0:
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(log_level)
        logger.addHandler(stream_handler)
        logger.setLevel(log_level)
    else:
        stream_handler.setLevel(log_level)
        logger.addHandler(stream_handler)
        logger.setLevel(log_level)
    # _accquire_lock()
    logger_initialized[name] = True
    # _release_lock()
    return logger
