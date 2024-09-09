import logging


def base_logger(file_path=None, mode='w'):
    logger = logging.getLogger(__name__)
    logger.setLevel('INFO')

    formatter = logging.Formatter(
        fmt='%(asctime)s %(process)d %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    if len(logger.handlers) > 0:
        logger.handlers[0].setFormatter(formatter)
    else:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    if file_path is not None:
        fh = logging.FileHandler(file_path, mode=mode)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


LOGGER = base_logger()
