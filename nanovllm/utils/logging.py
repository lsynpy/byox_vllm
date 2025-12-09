import inspect
import logging
import sys
import threading
from collections.abc import Hashable
from typing import Any


def make_hashable(obj: Any) -> Hashable:
    # Handle PyTorch tensors by converting to string representations of their shape and dtype
    if "torch" in sys.modules:
        import torch

        if isinstance(obj, torch.Tensor):
            return ("torch.Tensor", tuple(obj.shape), str(obj.dtype), obj.device.type)

    if isinstance(obj, list):
        return tuple(make_hashable(item) for item in obj)
    elif isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, set):
        return frozenset(make_hashable(item) for item in obj)
    elif isinstance(obj, (tuple, frozenset)):
        return tuple(make_hashable(item) for item in obj)
    else:
        return obj


_once_cache = set()
_once_lock = threading.Lock()


def _get_correct_caller_info(skip_files=None):
    if skip_files is None:
        skip_files = {"logging.py", "logger.py", "<string>", "torch/nn/modules/module.py"}

    frame = inspect.currentframe()
    # Go back two frames to skip _get_correct_caller_info and the calling logging method
    frame = frame.f_back.f_back if frame and frame.f_back else None

    while frame:
        filename = frame.f_code.co_filename.split("/")[-1]
        if filename not in skip_files:
            return frame.f_lineno, frame.f_code.co_filename
        frame = frame.f_back

    # If we can't find an appropriate frame, fall back to stacklevel
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


def debug_once(logger: logging.Logger, msg: str, *args, **kwargs):
    _log_once(logging.DEBUG, logger.name, msg, *args)


def info_once(logger: logging.Logger, msg: str, *args, **kwargs):
    _log_once(logging.INFO, logger.name, msg, *args)


def set_default_log_level(level: int):
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)


class LoggerWithOnceMethods:
    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def debug_once(self, msg: str, *args, **kwargs):
        _log_once(logging.DEBUG, self._logger.name, msg, *args)

    def info_once(self, msg: str, *args, **kwargs):
        _log_once(logging.INFO, self._logger.name, msg, *args)

    def debug(self, msg: str, *args, **kwargs):
        if self._logger.isEnabledFor(logging.DEBUG):
            lineno, filename = _get_correct_caller_info()
            if lineno is not None and filename is not None:
                record = self._logger.makeRecord(
                    self._logger.name, logging.DEBUG, filename, lineno, msg, args, None
                )
                return self._logger.handle(record)
        return self._logger.debug(msg, *args, stacklevel=2, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        if self._logger.isEnabledFor(logging.INFO):
            lineno, filename = _get_correct_caller_info()
            if lineno is not None and filename is not None:
                record = self._logger.makeRecord(
                    self._logger.name, logging.INFO, filename, lineno, msg, args, None
                )
                return self._logger.handle(record)
        return self._logger.info(msg, *args, stacklevel=2, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        if self._logger.isEnabledFor(logging.WARNING):
            lineno, filename = _get_correct_caller_info()
            if lineno is not None and filename is not None:
                record = self._logger.makeRecord(
                    self._logger.name, logging.WARNING, filename, lineno, msg, args, None
                )
                return self._logger.handle(record)
        return self._logger.warning(msg, *args, stacklevel=2, **kwargs)

    def warn(self, msg: str, *args, **kwargs):
        if self._logger.isEnabledFor(logging.WARNING):
            lineno, filename = _get_correct_caller_info()
            if lineno is not None and filename is not None:
                record = self._logger.makeRecord(
                    self._logger.name, logging.WARNING, filename, lineno, msg, args, None
                )
                return self._logger.handle(record)
        return self._logger.warning(msg, *args, stacklevel=2, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        if self._logger.isEnabledFor(logging.ERROR):
            lineno, filename = _get_correct_caller_info()
            if lineno is not None and filename is not None:
                record = self._logger.makeRecord(
                    self._logger.name, logging.ERROR, filename, lineno, msg, args, None
                )
                return self._logger.handle(record)
        return self._logger.error(msg, *args, stacklevel=2, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        if self._logger.isEnabledFor(logging.CRITICAL):
            lineno, filename = _get_correct_caller_info()
            if lineno is not None and filename is not None:
                record = self._logger.makeRecord(
                    self._logger.name, logging.CRITICAL, filename, lineno, msg, args, None
                )
                return self._logger.handle(record)
        return self._logger.critical(msg, *args, stacklevel=2, **kwargs)

    def log(self, level: int, msg: str, *args, **kwargs):
        if self._logger.isEnabledFor(level):
            lineno, filename = _get_correct_caller_info()
            if lineno is not None and filename is not None:
                record = self._logger.makeRecord(
                    self._logger.name, level, filename, lineno, msg, args, None
                )
                return self._logger.handle(record)
        return self._logger.log(level, msg, *args, stacklevel=2, **kwargs)

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


def get_logger(name: str, level: int = logging.INFO, use_color: bool = True) -> LoggerWithOnceMethods:
    if not logging.getLogger().handlers:
        handler = logging.StreamHandler(sys.stdout)

        if use_color:
            formatter = ColoredCombinedFormatter(
                fmt="%(asctime)s - %(combined_info)-21s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
            )
        else:
            formatter = CombinedFormatter(
                fmt="%(asctime)s - %(combined_info)-21s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
            )

        handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.NOTSET)

    logger = logging.getLogger(name)

    logger.setLevel(level)
    return LoggerWithOnceMethods(logger)


class CombinedFormatter(logging.Formatter):
    def format(self, record):
        combined = f"{record.filename}:{record.lineno}"
        record.combined_info = f"{combined:<20}"
        return super().format(record)


class ColoredCombinedFormatter(CombinedFormatter):
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset to default
    }

    def format(self, record):
        combined = f"{record.filename}:{record.lineno}"
        record.combined_info = f"{combined:<20}"

        level_color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset_color = self.COLORS["RESET"]
        colored_levelname = f"{level_color}{record.levelname:<5}{reset_color}"

        original_levelname = record.levelname
        record.levelname = colored_levelname
        formatted_message = super().format(record)
        record.levelname = original_levelname

        return formatted_message
