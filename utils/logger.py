import logging
import os
from typing import Dict, Optional
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

class CudaLogger:
    _instance = None
    _loggers: Dict[str, logging.Logger] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CudaLogger, cls).__new__(cls)
            cls._instance._configure_root_logger()
        return cls._instance

    def _configure_root_logger(self):
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        # File handler
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=os.path.join(log_dir, "cuda_to_metal.log"),
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    def get_logger(self, name: str) -> logging.Logger:
        if name not in self._loggers:
            logger = logging.getLogger(name)
            self._loggers[name] = logger
        return self._loggers[name]

    def set_log_level(self, level: int):
        for logger in self._loggers.values():
            logger.setLevel(level)

    def add_file_handler(self, filename: str, level: int = logging.DEBUG,
                         max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5):
        file_handler = RotatingFileHandler(
            filename=filename,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
        file_handler.setFormatter(formatter)
        for logger in self._loggers.values():
            logger.addHandler(file_handler)

    def add_timed_rotating_file_handler(self, filename: str, level: int = logging.DEBUG,
                                        when: str = 'midnight', interval: int = 1, backup_count: int = 7):
        file_handler = TimedRotatingFileHandler(
            filename=filename,
            when=when,
            interval=interval,
            backupCount=backup_count
        )
        file_handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
        file_handler.setFormatter(formatter)
        for logger in self._loggers.values():
            logger.addHandler(file_handler)

def get_logger(name: str) -> logging.Logger:
    return CudaLogger().get_logger(name)

# Convenience functions for different log levels
def debug(logger: logging.Logger, message: str, *args, **kwargs):
    logger.debug(message, *args, **kwargs)

def info(logger: logging.Logger, message: str, *args, **kwargs):
    logger.info(message, *args, **kwargs)

def warning(logger: logging.Logger, message: str, *args, **kwargs):
    logger.warning(message, *args, **kwargs)

def error(logger: logging.Logger, message: str, *args, **kwargs):
    logger.error(message, *args, **kwargs)

def critical(logger: logging.Logger, message: str, *args, **kwargs):
    logger.critical(message, *args, **kwargs)

def exception(logger: logging.Logger, message: str, *args, exc_info=True, **kwargs):
    logger.exception(message, *args, exc_info=exc_info, **kwargs)

# Performance logging
def log_performance(logger: logging.Logger, operation: str, execution_time: float):
    logger.info(f"Performance: {operation} took {execution_time:.4f} seconds")

# Function entry/exit logging
def log_function_entry(logger: logging.Logger, func_name: str, args: Optional[Dict] = None):
    args_str = ", ".join(f"{k}={v}" for k, v in args.items()) if args else ""
    logger.debug(f"Entering function: {func_name}({args_str})")

def log_function_exit(logger: logging.Logger, func_name: str, result: Any = None):
    logger.debug(f"Exiting function: {func_name} with result: {result}")

# Context manager for function logging
class LogFunction:
    def __init__(self, logger: logging.Logger, func_name: str):
        self.logger = logger
        self.func_name = func_name

    def __enter__(self):
        log_function_entry(self.logger, self.func_name)

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            self.logger.exception(f"Exception in function {self.func_name}: {exc_value}")
        else:
            log_function_exit(self.logger, self.func_name)