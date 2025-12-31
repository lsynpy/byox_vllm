import inspect
import logging
import sys
import threading
from collections.abc import Hashable
from typing import Any

import torch

# _FORMAT = "%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s"
_FORMAT = "%(message)s"


def make_hashable(obj: Any) -> Hashable:
    if "torch" in sys.modules:
        import torch

        if isinstance(obj, torch.Tensor):
            return ("torch.Tensor", tuple(obj.shape), str(obj.dtype), obj.device.type)

    if isinstance(obj, (list, tuple)):
        return tuple(make_hashable(item) for item in obj)
    elif isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, set):
        return frozenset(make_hashable(item) for item in obj)
    else:
        return obj


_once_cache = set()
_once_lock = threading.Lock()


def _get_correct_caller_info(skip_files=None):
    if skip_files is None:
        skip_files = {"logging.py", "logger.py", "<string>", "torch/nn/modules/module.py"}

    frame = inspect.currentframe()
    if frame and frame.f_back:
        frame = frame.f_back.f_back

    while frame:
        filename = frame.f_code.co_filename.split("/")[-1]
        if filename not in skip_files:
            return frame.f_lineno, frame.f_code.co_filename
        frame = frame.f_back

    return None, None


def _log_once(level: int, logger_name: str, msg: str, *args) -> None:
    hashable_args = tuple(make_hashable(arg) for arg in args)
    cache_key = (logger_name, msg, hashable_args, level)

    with _once_lock:
        if cache_key in _once_cache:
            return
        _once_cache.add(cache_key)

    logger = logging.getLogger(logger_name)
    lineno, filename = _get_correct_caller_info()
    if lineno is not None and filename is not None:
        record = logger.makeRecord(logger.name, level, filename, lineno, msg, args, None)
        logger.handle(record)
    else:
        logger.log(level, msg, *args, stacklevel=2)


def set_default_log_level(level: int):
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)


class LoggerWithOnceMethods:
    def __init__(self, logger: logging.Logger):
        self._logger = logger
        self.debug_once = lambda msg, *args, **kwargs: _log_once(
            logging.DEBUG, self._logger.name, msg, *args
        )
        self.info_once = lambda msg, *args, **kwargs: _log_once(
            logging.INFO, self._logger.name, msg, *args
        )

    def _log_with_location(self, level: int, msg: str, *args, **kwargs):
        if self._logger.isEnabledFor(level):
            lineno, filename = _get_correct_caller_info()
            if lineno is not None and filename is not None:
                record = self._logger.makeRecord(
                    self._logger.name, level, filename, lineno, msg, args, None
                )
                return self._logger.handle(record)
        return self._logger.log(level, msg, *args, stacklevel=2, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        return self._log_with_location(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        return self._log_with_location(logging.INFO, msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        return self._log_with_location(logging.ERROR, msg, *args, **kwargs)

    def log(self, level: int, msg: str, *args, **kwargs):
        return self._log_with_location(level, msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs):
        return self._logger.exception(msg, *args, **kwargs)

    @property
    def name(self):
        return self._logger.name

    @property
    def level(self):
        return self._logger.level

    @property
    def parent(self):
        return self._logger.parent

    @property
    def propagate(self):
        return self._logger.propagate

    @property
    def handlers(self):
        return self._logger.handlers

    @property
    def disabled(self):
        return self._logger.disabled


class ColoredFileNameLineFormatter(logging.Formatter):
    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "ERROR": "\033[31m",  # Red
    }
    RESET = "\033[0m"  # Reset to default

    def format(self, record):
        levelname = record.levelname
        colored_levelname = self.COLORS.get(levelname, "") + levelname + self.RESET
        record.levelname = colored_levelname

        formatter = logging.Formatter(fmt=_FORMAT, datefmt="%H:%M:%S")
        msg = formatter.format(record)

        if "\n" in msg:
            prefix_end = msg.find(record.getMessage()) if record.getMessage() else msg.rfind("]") + 1
            if prefix_end != -1:
                prefix = msg[:prefix_end]
                message_lines = msg[prefix_end:].split("\n")
                aligned_msg = (
                    prefix
                    + message_lines[0]
                    + "".join(f"\n{prefix}{line}" for line in message_lines[1:])
                )
                msg = aligned_msg

        record.levelname = levelname
        return msg


# Global counter for tensor saving
_tensor_save_counter = 0


def save_tensor(tensor, filename_prefix: str = "tensor"):
    global _tensor_save_counter
    _tensor_save_counter += 1
    filename = f"{_tensor_save_counter}_{filename_prefix}.pt"
    torch.save(tensor, filename)


def get_logger(name: str, level: int = logging.INFO) -> LoggerWithOnceMethods:
    if not logging.getLogger().handlers:
        handler = logging.StreamHandler(sys.stdout)

        formatter = ColoredFileNameLineFormatter()

        handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.NOTSET)

    logger = logging.getLogger(name)

    logger.setLevel(level)
    return LoggerWithOnceMethods(logger)
