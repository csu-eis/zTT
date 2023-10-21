import colorlog
import logging
import os

def formatted_time():
    from datetime import datetime
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    return formatted_time

def get_logger(prefix=""):
    
    logger = logging.getLogger(f'zTT-{prefix}')
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    fmt_string = '%(log_color)s[%(asctime)s][%(name)s][%(levelname)s]%(message)s'
    # black red green yellow blue purple cyan 和 white
    log_colors = {
        'DEBUG': 'white',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'purple'
        }
    fmt = colorlog.ColoredFormatter(fmt_string, datefmt="%Y-%m-%d %H:%M:%S", log_colors=log_colors)
    stream_handler.setFormatter(fmt)
    os.makedirs("tmp",exist_ok=True)
    file_handler = logging.FileHandler(f"tmp/{prefix}-log-{formatted_time()}.txt")
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger
    # return logger
        