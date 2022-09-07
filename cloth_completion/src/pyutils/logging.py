import logging

_logger = None


def setup():
    logging.basicConfig(
        format="[%(asctime)s][%(levelname)s][%(name)s]:%(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_logger():
    """Returns a global logger singletone."""
    global _logger

    if _logger:
        return _logger

    _logger = logging.getLogger("codec_hand")

    logging.basicConfig(
        format="[%(asctime)s][%(levelname)s][%(name)s]:%(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    return _logger
