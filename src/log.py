import logging
import sys

# Create a custom logger
logger = logging.getLogger("custom_logger")

# Set the default log level (INFO can be adjusted to DEBUG, WARNING, etc., as needed)
logger.setLevel(logging.INFO)

# Create a console handler for logging to standard output
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Define a log message format
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(log_format)

# Attach the formatter to the console handler
console_handler.setFormatter(formatter)

# Add the console handler to the logger
logger.addHandler(console_handler)

# Convenience logging functions
def info(message: str):
    """Log an info-level message."""
    logger.info(message)

def error(message: str):
    """Log an error-level message."""
    logger.error(message)

def warning(message: str):
    """Log a warning-level message."""
    logger.warning(message)

def debug(message: str):
    """Log a debug-level message."""
    logger.debug(message)