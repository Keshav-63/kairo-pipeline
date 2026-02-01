# kairo_pipeline/config/logger.py
import logging
import os
from logging.handlers import RotatingFileHandler

# Define log directory and file
log_dir = os.path.join(os.path.dirname(__file__), '../logs')
os.makedirs(log_dir, exist_ok=True)  # Create logs directory if it doesn't exist
log_file = os.path.join(log_dir, 'pipeline.log')

# Configure logger
logger = logging.getLogger('kairo_pipeline')
logger.setLevel(logging.DEBUG)

# Set up rotating file handler
handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info("Logger initialized")