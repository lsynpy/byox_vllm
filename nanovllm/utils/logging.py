import logging
import sys


def init_logger(name: str | None = None, level: int = logging.DEBUG) -> logging.Logger:
    # Set up basic logging configuration if not already configured
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stdout,
        )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    return logger


def set_logger_level(logger: logging.Logger, level: int):
    logger.setLevel(level)


def set_global_log_level(level: int):
    logging.getLogger().setLevel(level)
    # Also set the global basicConfig level
    for handler in logging.getLogger().handlers:
        handler.setLevel(level)


# Global logger instance for the package
logger = init_logger(__name__.split(".")[0])  # Get module name without submodules
