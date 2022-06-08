import warnings
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from PIL import Image, ImageFilter
from cv2 import imread as cv2_imread
from pco_tools import pco_reader as pco


def scale(img: np.ndarray, max: float) -> np.ndarray:
    """scales image to new max"""
    _min = np.min(img)
    _max = np.max(img)
    return (img - _min) / (_max - _min) * max


def scale_to_8bit(img: np.ndarray) -> np.ndarray:
    """scales image to 8 bit"""
    return scale(img, 2 ** 8)


def normalize_to_min_4sigma(img):
    """normalizes the image to interval [0, 4*sigma] as proposed in 'Autocorrelation-based estimate of
    PIV density for diffraction limited particle images'

    Min value will be 0 and max will be mu+4*sigma. Value above will be saturated
    """
    if img.ndim == 2:
        _min = np.min(img)
        _img = img - _min
        _mu = np.mean(_img)
        _std = np.std(_img)
        _max = _mu + 4 * _std
        _img[_img > _max] = _max
        _img[_img < 0] = 0
    elif img.ndim == 3:
        _min = np.min(img, axis=0)
        _img = img - _min
        _mu = np.mean(_img, axis=(1, 2))
        _std = np.std(_img, axis=(1, 2))
        _max = _mu + 4 * _std
        for i in range(_img.shape[0]):
            _img[i][_img[i] > _max[i]] = _max[i]
            _img[i][_img[i] < 0] = 0
    else:
        raise ValueError(f'Wrong image dimension. Must be 2D or 3D, but not {img.ndim}D.')
    return _img


def pivhist(img: Union[np.ndarray, xr.DataArray], bins=100, **kwargs) -> plt.Axes:
    ax = kwargs.pop('ax', None)
    bins = kwargs.pop('bins', bins)
    if ax is None:
        ax = plt.gca()

    if img.ndim != 2:
        raise ValueError(f'Image must be 2D, not {img.ndim}D.')
    if isinstance(img, xr.DataArray):
        _ = img[:, :].plot.hist(bins=bins, density=True, **kwargs)
    else:
        plt.hist(img.flatten(), bins=bins, density=True, **kwargs)
        ax.set_xlabel('intensity / counts')
    ax.set_yscale('log')
    ax.set_ylabel('density')
    return ax


def load_img(img_filepath: Path):
    """
    loads b16 or other file format
    """
    img_filepath = Path(img_filepath)
    if not img_filepath.exists():
        raise FileExistsError(f'Image "{img_filepath}" not found.')

    if img_filepath.suffix == '.b16':
        return pco.load(str(img_filepath))

    return cv2_imread(str(img_filepath), -1)


def subtract_background(img: np.ndarray, method: str, axis: int = None) -> np.ndarray:
    bg = compute_background(img, method, axis)
    if img.ndim == 2:
        return img - bg
    else:
        np.subtract(img, bg, axis=axis)


def compute_background(img: np.ndarray, method: str, axis: int = None) -> np.ndarray:
    """Computes a background image based on the image input"""
    if img.ndim == 1:
        raise ValueError('Invalid img array dimension (1D). Must be 2D or 3D.')
    if img.ndim == 2:
        if axis is not None:
            warnings.warn(f'Image dimension is 2D and axis={axis} was specified. '
                          'This is an invalid combination of parameters. '
                          'Axis will be ignored')
    elif img.ndim > 3:
        raise ValueError(f'Invalid img array dimension ({img.ndim}D). Must be 2D or 3D.')
    if method == 'random':
        # if isinstance(img, xr.DataArray):
        #     img = img.values()
        bg = np.asarray(img.copy())
        np.random.shuffle(bg)
        return bg
    elif method == 'min':
        return np.min(img, axis=axis)
    else:
        raise ValueError(f'Invalid method: {method}')


def source_density_2d(particle_diameter: float, image_size: float) -> float:
    """the source density in PIV is defined for volumetric PIV, however, for 2D PIV
    it could also be a useful measure

    Parameters
    ----------
    particle_diameter: float
        Particle diameter in [px]
    image_size: float
        Image size (width*height) in [px]

    Returns
    -------
    Image particle area divided by the image size
    """
    return np.pi * particle_diameter ** 2 / 4 / image_size


def min_max_filter(img, filter_size: int):
    """returns the min-max-filtered image

    Parameters
    ----------
    img: np.ndarray
        Input image
    filter_size: int
        Filter size. Must be odd.

    Returns
    -------
    Min-Max-filtered image
    """
    _img = Image.fromarray(img)
    img_min = np.array(_img.filter(ImageFilter.MinFilter(size=filter_size)))
    img_max = np.array(_img.filter(ImageFilter.MaxFilter(size=filter_size)))
    return (np.array(img) - img_min) / (img_max - img_min)


def min_median_filter(img: np.ndarray, filter_size: int) -> np.ndarray:
    """returns the min-max-filtered image

    Parameters
    ----------
    img: np.ndarray
        Input image
    filter_size: int
        Filter size. Must be odd.

    Returns
    -------
    Min-Max-filtered image
    """
    _img = Image.fromarray(img)
    img_min = np.array(_img.filter(ImageFilter.MinFilter(size=filter_size)))
    img_median = np.array(_img.filter(ImageFilter.MedianFilter(size=filter_size)))
    return (np.array(img) - img_min) / (img_median - img_min)


def min_filter(img: np.ndarray, filter_size: int) -> np.ndarray:
    """returns the min filtered img"""
    _img = Image.fromarray(img)
    return np.array(_img.filter(ImageFilter.MinFilter(size=filter_size)))


def max_filter(img: np.ndarray, filter_size: int) -> np.ndarray:
    """returns the max filtered img"""
    _img = Image.fromarray(img)
    return np.array(_img.filter(ImageFilter.MaxFilter(size=filter_size)))


def median_filter(img: np.ndarray, filter_size: int) -> np.ndarray:
    """returns the median filtered img"""
    _img = Image.fromarray(img)
    return np.array(_img.filter(ImageFilter.MedianFilter(size=filter_size)))


def min_max_background_subtraction(img1, img2, filter_size):
    """applies min-max-background image on input images"""
    min_max_diff = min_max_filter(img1, filter_size) - min_max_filter(img2, filter_size)
    img1r = np.zeros_like(min_max_diff)
    flag_1 = min_max_diff > 0
    flag_2 = min_max_diff < 0
    img1r[flag_1] = min_max_diff[flag_1]
    img2r = np.zeros_like(min_max_diff)
    img2r[flag_2] = np.abs(min_max_diff[flag_2])
    return img1r, img2r
