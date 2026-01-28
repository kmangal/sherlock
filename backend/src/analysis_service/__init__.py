import logging
import sys

# 1. Set up a handler and formatter (e.g., for console output)
# This handler will be used by all loggers that don't have their own handlers.
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(
    logging.DEBUG
)  # The handler should process all messages
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(formatter)

# 2. Get the root logger and set its level to DEBUG
# All application loggers (using __name__) will inherit from the root logger by default.
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(console_handler)

# 3. Suppress specific chatty library loggers by setting their levels higher
# If needed
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
