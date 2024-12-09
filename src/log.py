import logging
import sys

# Create a custom logger
logger = logging.getLogger("logger")

# Set the default log level (can be adjusted as needed)
logger.setLevel(logging.INFO)

# Create handlers
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Create a formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(console_handler)

# Convenience methods for logging
def info(message):
    logger.info(message)

def error(message):
    logger.error(message)

def warning(message):
    logger.warning(message)

def debug(message):
    logger.debug(message)
