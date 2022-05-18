from pathlib import Path

from cv2 import imread as cv2_imread
from pco_tools import pco_reader as pco


def loadimg(img_filepath: Path):
    """
    loads b16 or other file format
    """
    img_filepath = Path(img_filepath)
    if not img_filepath.exists():
        raise FileExistsError(f'Image "{img_filepath}" not found.')

    if img_filepath.suffix == '.b16':
        return pco.load(str(img_filepath))

    return cv2_imread(str(img_filepath), -1)
