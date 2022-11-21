import logging as logger
import os
import sys
from logging.handlers import RotatingFileHandler

LOG_DIR = 'Log'
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
handlers = [
        RotatingFileHandler(os.path.join(LOG_DIR,'log.log'), maxBytes=1_000_000, backupCount=10),            
        logger.StreamHandler(sys.stdout)            
    ]
logger.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',handlers=handlers, level=logger.INFO)

def get_logger(name:str) -> logger.Logger:
    return logger.getLogger(name)


