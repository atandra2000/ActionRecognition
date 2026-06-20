"""
Logging utilities for the action recognition system
"""

import os
import sys
import logging
import logging.handlers
from typing import Optional, Dict, Any
from contextlib import contextmanager
from contextvars import ContextVar
import time

# Module-level constants
DEFAULT_FORMATTER = logging.Formatter(
    '[%(asctime)s] [%(levelname)s] [%(name)s] [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
VALID_LOG_LEVELS = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
_correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
_logger_cache: Dict[str, logging.Logger] = {}  # Cache to prevent duplicate initialization


def _validate_log_level(level: str) -> str:
    """Validate and normalize log level string.
    
    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Normalized uppercase log level string
        
    Raises:
        ValueError: If log level is invalid
    """
    normalized = level.upper()
    if normalized not in VALID_LOG_LEVELS:
        raise ValueError(f"Invalid log level: {level}. Must be one of {VALID_LOG_LEVELS}")
    return normalized


def _clear_logger_handlers(logger: logging.Logger) -> None:
    """Remove all handlers from a logger.
    
    Args:
        logger: Logger instance to clear
    """
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def reset_logger(name: str) -> None:
    """Reset logger configuration, clearing all handlers.
    
    Args:
        name: Logger name
    """
    logger = logging.getLogger(name)
    _clear_logger_handlers(logger)
    if name in _logger_cache:
        del _logger_cache[name]


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    formatter: Optional[logging.Formatter] = None
) -> logging.Logger:
    """
    Setup logger with both file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        formatter: Custom formatter (uses default if not provided)
    
    Returns:
        Configured logger instance
        
    Raises:
        ValueError: If log level is invalid
    """
    # Validate log level
    level = _validate_log_level(level)
    
    # Use default formatter if not provided
    if formatter is None:
        formatter = DEFAULT_FORMATTER
    
    # Create logger (or get existing and clear handlers)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    # Clear any existing handlers to prevent duplicates
    _clear_logger_handlers(logger)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Create directory if needed
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent duplicate handlers through propagation
    logger.propagate = False
    
    # Cache logger
    _logger_cache[name] = logger
    
    return logger


def get_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Get existing logger from cache or create a new one.
    
    Args:
        name: Logger name
        log_level: Logging level
    
    Returns:
        Logger instance from cache or newly created
        
    Raises:
        ValueError: If log level is invalid
    """
    # Return cached logger if available
    if name in _logger_cache:
        return _logger_cache[name]
    
    # Create and cache new logger
    return setup_logger(name, level=log_level)


class LoggerMixin:
    """Mixin class to add logging functionality.
    
    WARNING: Requires cooperative multiple inheritance.
    All classes in the MRO must call super().__init__().
    Do not use with classes that override __getattr__.
    """
    
    def __init__(self, *args, logger_name: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        # Use custom name or auto-generate from class module and name
        if logger_name:
            self.logger = get_logger(logger_name)
        else:
            logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
            self.logger = get_logger(logger_name)
    
    def __getattr__(self, name: str):
        """Dynamically route log_* methods to logger methods."""
        if name.startswith('log_'):
            level = name[4:].upper()  # Extract level from log_debug -> DEBUG
            if level in VALID_LOG_LEVELS:
                return getattr(self.logger, level.lower())
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID for request tracking.
    
    Args:
        correlation_id: Unique identifier for tracking related log messages
    """
    _correlation_id.set(correlation_id)


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID.
    
    Returns:
        Current correlation ID or None
    """
    return _correlation_id.get()


@contextmanager
def log_timed_block(logger: logging.Logger, block_name: str, level: str = "INFO"):
    """Context manager for timed logging blocks.
    
    Args:
        logger: Logger instance
        block_name: Name of the code block
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Yields:
        Dictionary with timing information
        
    Example:
        with log_timed_block(logger, "data_loading") as timer:
            load_data()  # Automatically logs execution time
    """
    level = _validate_log_level(level)
    log_func = getattr(logger, level.lower())
    
    start_time = time.time()
    log_func(f"Starting: {block_name}")
    
    timing_info = {'start': start_time}
    
    try:
        yield timing_info
    finally:
        elapsed = time.time() - start_time
        timing_info['elapsed'] = elapsed
        log_func(f"Completed: {block_name} (took {elapsed:.2f}s)")


def log_system_info(logger: logging.Logger):
    """Log system information with proper error handling.
    
    Args:
        logger: Logger instance
    """
    import platform
    
    try:
        import torch
        pytorch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
    except ImportError:
        pytorch_version = "Not installed"
        cuda_available = False
    
    logger.info("=" * 50)
    logger.info("System Information")
    logger.info("=" * 50)
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"PyTorch: {pytorch_version}")
    logger.info(f"CUDA Available: {cuda_available}")
    
    # Safely log GPU information only if available
    if cuda_available:
        try:
            cuda_version = torch.version.cuda
            device_name = torch.cuda.get_device_name(0)
            device_props = torch.cuda.get_device_properties(0)
            gpu_memory = device_props.total_memory / 1e9
            
            logger.info(f"CUDA Version: {cuda_version}")
            logger.info(f"GPU: {device_name}")
            logger.info(f"GPU Memory: {gpu_memory:.2f} GB")
        except Exception as e:
            logger.warning(f"Failed to retrieve GPU details: {e}")
    
    logger.info("=" * 50)
    