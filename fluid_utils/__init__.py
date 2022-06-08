from . import piv
from ._logger import logger, _file_handler, _stream_handler

from ._version import __version__

__all__ = [__version__]

def set_loglevel(level):
    logger.setLevel(level.upper())
    _file_handler.setLevel(level.upper())
    _stream_handler.setLevel(level.upper())
