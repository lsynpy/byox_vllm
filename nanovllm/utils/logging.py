import logging
import sys


def set_default_log_level(level: int):
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)


def get_logger(name: str, level: int | None = logging.INFO, use_color: bool = True) -> logging.Logger:
    if not logging.getLogger().handlers:
        # Add a custom formatter to combine filename and lineno
        handler = logging.StreamHandler(sys.stdout)

        if use_color:
            formatter = ColoredCombinedFormatter(
                fmt="%(asctime)s - %(combined_info)-20s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
            )
        else:
            formatter = CombinedFormatter(
                fmt="%(asctime)s - %(combined_info)-20s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
            )

        handler.setFormatter(formatter)

        # Set up the root logger with our custom handler
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)

    logger = logging.getLogger(name)

    if level is not None:
        logger.setLevel(level)

    return logger


class CombinedFormatter(logging.Formatter):
    def format(self, record):
        combined = f"{record.filename}:{record.lineno}"
        record.combined_info = f"{combined:<20}"
        return super().format(record)


class ColoredCombinedFormatter(CombinedFormatter):
    # Define color codes for different log levels
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset to default
    }

    def format(self, record):
        # Set the combined_info attribute first (like parent class does)
        combined = f"{record.filename}:{record.lineno}"
        record.combined_info = f"{combined:<20}"

        # Apply color to the level name and also add proper padding
        level_color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset_color = self.COLORS["RESET"]
        # Add padding to colored level name to maintain alignment
        colored_levelname = f"{level_color}{record.levelname:<5}{reset_color}"

        # Temporarily store original levelname and replace with colored version
        original_levelname = record.levelname
        record.levelname = colored_levelname

        # Get the formatted string with colored level
        formatted_message = super().format(record)

        # Restore original levelname
        record.levelname = original_levelname

        return formatted_message
