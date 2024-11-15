from logging.handlers import RotatingFileHandler
from datetime import datetime
import os
import logging
import datetime


def get_logger(name):
    """Get a logger with the specified name"""
    logger = logging.getLogger(name)
    # Don't propagate to root logger to avoid duplicate messages
    logger.propagate = False
    return logger

def setup_logging(log_level):
    """Setup logging with minimal debug output"""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        # Create logs directory if it doesn't exist
        log_dir = os.path.join(os.getcwd(), 'logs')
        os.makedirs(log_dir, exist_ok=True)

    # Generate timestamp for log file
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'hids_{timestamp}.log')

    # Configure logging format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)

    # Setup console handler with less verbose output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)  # Only show INFO and above in console

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Disable matplotlib debug logging
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    # Disable other verbose loggers
    logging.getLogger('PIL').setLevel(logging.WARNING)


class ColoredFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning/errors."""

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)



class PerformanceLogger:
    """
    Specialized logger for performance metrics
    """
    def __init__(self):
        self.logger = logging.getLogger('performance')

    def log_metrics(self, metrics):
        """
        Log performance metrics
        """
        self.logger.info(
            "Performance Metrics - "
            f"CPU: {metrics.get('cpu', 0):.2f}% | "
            f"Memory: {metrics.get('memory', 0):.2f}MB | "
            f"Detection Time: {metrics.get('detection_time', 0):.4f}s"
        )

    def log_anomaly(self, details):
        """
        Log anomaly detection details
        """
        self.logger.warning(
            "Anomaly Detected - "
            f"Process: {details.get('process', 'Unknown')} | "
            f"Score: {details.get('score', 0):.4f} | "
            f"Threshold: {details.get('threshold', 0):.4f}"
        )