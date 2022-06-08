import logging
import pathlib
import time
from typing import Tuple

import numpy as np
import scipy.interpolate as interp
import scipy.ndimage.filters as filt

from .image import load_img

_formatter = logging.Formatter(
    '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d_%H:%M:%S')

logger = logging.getLogger(__package__)

try:
    from numba import jit

    JIT_EXISTS = True
except ImportError:
    JIT_EXISTS = False
    logger.info('Numba could not be imported. Some computations may run slower without the package')

# -------------
# Following code is taken from and adapted/improved further: https://github.com/OpenPIV/piv_uncertainty
# improvement especially through speed up through jit and clear documentation of array units!
# before publication of this code, ask author or write code yourself!
# TODO: is overlap needed in correlation_stats?!

"""
PIV ERROR ANALYSIS FUNCTION


Cameron Dallas
University of Toronto
Department of Mechanical and Industrial Engineering
Turbulence Research Laboratory
"""

if JIT_EXISTS:
    @jit(nopython=True)
    def _smooth_dC(dC: np.ndarray) -> np.ndarray:
        # smooth dC_x in the x direction
        for i in range(dC.shape[0]):
            for j in range(1, dC.shape[1] - 1):
                dC[i, j] = (dC[i, j - 1] + 2 * dC[i, j] + dC[i, j + 1]) / 4.
        return dC
else:
    def _smooth_dC(dC: np.ndarray) -> np.ndarray:
        # smooth dC_x in the x direction
        for i in range(dC.shape[0]):
            for j in range(1, dC.shape[1] - 1):
                dC[i, j] = (dC[i, j - 1] + 2 * dC[i, j] + dC[i, j + 1]) / 4.
        return dC

if JIT_EXISTS:
    @jit(nopython=True)  # boundscheck=True --> use for debugging
    def _compute_covariance_matrix_S(frame_a: np.ndarray, dC: np.ndarray, S: np.ndarray) -> np.ndarray:
        for i in range(frame_a.shape[0]):
            for j in range(frame_a.shape[1]):
                S0 = dC[i, j]
                for k in range(1, 5):
                    # Avoid IndexError Exception; same effect as try-except in original code by C. Dallas
                    if i + k < frame_a.shape[0] and j + k < frame_a.shape[1]:
                        if dC[i + k, j + k] / S0 < 0.05:
                            S[i, j] = np.sum(S0 * dC[i:i + k, j:j + k])
                            break
                        if k == 4:
                            S[i, j] = np.sum(S0 * dC[i:i + k, j:j + k])
        return S
else:
    def _compute_covariance_matrix_S(frame_a: np.ndarray, dC: np.ndarray, S: np.ndarray) -> np.ndarray:
        for i in range(frame_a.shape[0]):
            for j in range(frame_a.shape[1]):
                S0 = dC[i, j]
                for k in range(1, 5):
                    # Avoid IndexError Exception; same effect as try-except in original code by C. Dallas
                    if i + k < frame_a.shape[0] and j + k < frame_a.shape[1]:
                        if dC[i + k, j + k] / S0 < 0.05:
                            S[i, j] = np.sum(S0 * dC[i:i + k, j:j + k])
                            break
                        if k == 4:
                            S[i, j] = np.sum(S0 * dC[i:i + k, j:j + k])
        return S


def correlation_stats(frame_a: np.ndarray, frame_b: np.ndarray,
                      x_px: np.ndarray, y_px: np.ndarray,
                      u_px: np.ndarray, v_px: np.ndarray,
                      L: int, dx: int = 1, dy: int = 1,
                      kx: int = 1, ky: int = 1,
                      return_debug_data: bool = False,
                      jit_acceleration: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uses the Correlation Statistics method to approximate the error in the velocity
    arising from the image pair correlation algorithm.
    See paper 'PIV uncertainty quantification from correlation statistics'. 2015

    Parameters
    ----------
    frame_a: 2d array
        first full frame
    frame_b: 2d array
        second full frame
    x_px: 2d array , float
        x location of velocity vectors in [px]
    y_px: 2d array, float
        location of velocity vectors in [px]
    u_px: 2d array , float
        x velocity in [px]
    v_px: 2d array , float
        y velocity in [px]
    L: int
        Interrogation window size
    scaling factor: float
        image scaling factor in pixels per meter. By default this assumes that all inputs for
        velocity and location are in units of pixels and pixels per second
    dx: int
        pixels displacement for the delta x value (typically should be 1)
    dy: int
        pixel displacement for the delta y value (typically should be 1)
    kx, ky : ints, optional
        Degrees of the bivariate spline. Default is 1.
    return_debug_data : `bool`, optional
        Whether to return the dewarped image and interpolated velocites (for each pixel)
        as third return-value.
        Default is False.

    Returns
    -------
    Ux_px: `2d array`
        x displacement uncertainty in px
    Uy_px: `2d array`
        y displacement uncertainty in px

    depending on `return_debug_data`:
    frame_b_shift : `array_like`
        dewarped image B
    u_interp_px : `2d array`
        pixel-wise x-displacement in px
    v_interp_px : `2d array`
        pixel-wise y-displacement in px

    """

    if isinstance(frame_a, (str, pathlib.Path)):
        frame_a = load_img(frame_a)
    else:
        frame_a = np.asarray(frame_a)
    if isinstance(frame_b, (str, pathlib.Path)):
        frame_b = load_img(frame_b)
    else:
        frame_b = np.asarray(frame_b)

    if frame_a.ndim != 2:
        raise ValueError(f'Frame A must be 2D but is {frame_a.ndim}D')
    if frame_b.ndim != 2:
        raise ValueError(f'Frame B must be 2D but is {frame_b.ndim}D')

    # Make sure inputs make sense
    if frame_a.shape != frame_b.shape:
        raise ValueError('Image pair must be the same shape')

    if x_px.ndim == 1 and y_px.ndim == 1:
        # build 2D array out of it
        _x_px, _y_px = np.meshgrid(x_px, y_px)
    else:
        _x_px = x_px
        _y_px = y_px

    if _x_px.ndim != 2:
        raise ValueError(f'Vector location array in pixel units "x_px" must be 2D but is {_x_px.ndim}D')
    if _y_px.ndim != 2:
        raise ValueError(f'Vector location array in pixel units "y_px" must be 2D but is {_y_px.ndim}D')
    if not (_x_px.shape == _y_px.shape == u_px.shape == v_px.shape):
        raise ValueError('Vector shapes and locations must be the same size')

    # normalize images/ convert to float
    frame_a = frame_a[:].astype(float)
    frame_b = frame_b[:].astype(float)

    row_pix = _y_px.astype('int32')
    col_pix = _x_px.astype('int32')

    # Dewarp frame_b
    # x, y must be in [pix]!
    st = time.perf_counter()
    frame_b_shift, u_interp_px, v_interp_px = image_dewarp(frame_b, _x_px, _y_px, u_px, v_px, kx=kx, ky=ky,
                                                           jit_acceleration=jit_acceleration)
    logger.debug(f'dewarping took {time.perf_counter() - st}')

    """
    -----------------  x uncertainty -----------------------

    Get C and Sxy values for each pixel then filter and smooth them, which is
    equivalent to doing the sums over the interrogation window.
    """

    # calculate C(u)
    C_u = np.multiply(frame_a, frame_b_shift)  # this value is the same for both x and y uncertainty

    # get delta C_xy for x direction
    # dC_x = np.zeros(frame_a.shape)  # delta C_i in x direction
    cPlus_x = np.zeros(frame_a.shape)
    cPlus_x[:, 0:-dx] = frame_a[:, 0:-dx] * frame_b_shift[:, dx:]
    cMinus_x = np.zeros(frame_a.shape)
    cMinus_x[:, 0:-dx] = frame_a[:, dx:] * frame_b_shift[:, 0:-dx]
    dC_x = cPlus_x - cMinus_x
    dC_x = dC_x - np.mean(dC_x)  # remove the mean

    # smooth dC_x in the x direction
    if jit_acceleration:
        dC_x = _smooth_dC(dC_x)
    else:
        for i in range(dC_x.shape[0]):
            for j in range(1, dC_x.shape[1] - 1):
                dC_x[i, j] = (dC_x[i, j - 1] + 2 * dC_x[i, j] + dC_x[i, j + 1]) / 4.

    # calculate covariance matrix (S_x)
    S_x = np.zeros(frame_a.shape)
    if jit_acceleration:
        S_x = _compute_covariance_matrix_S(frame_a, dC_x, S_x)
    else:
        for i in range(frame_a.shape[0]):
            for j in range(frame_a.shape[1]):

                S0 = dC_x[i, j]
                for k in range(1, 5):
                    try:
                        if dC_x[i + k, j + k] / S0 < 0.05:
                            S_x[i, j] = np.sum(S0 * dC_x[i:i + k, j:j + k])
                            break
                        if k == 4:
                            S_x[i, j] = np.sum(S0 * dC_x[i:i + k, j:j + k])
                    except IndexError:
                        S_x[i, j] = 0.

    # smooth and sum the fields
    C_filt = (L ** 2) * filt.gaussian_filter(C_u, L)[row_pix, col_pix]
    cPlus_filt_x = (L ** 2) * filt.gaussian_filter(cPlus_x, L)[row_pix, col_pix]
    cMinus_filt_x = (L ** 2) * filt.gaussian_filter(cMinus_x, L)[row_pix, col_pix]
    S_x_filt = (L ** 2) * filt.gaussian_filter(S_x, L)[row_pix, col_pix]

    sig_x = np.sqrt(S_x_filt)
    cpm_x = (cPlus_filt_x + cMinus_filt_x) / 2.

    # final x uncertainty
    Ux_px = ((np.log(cpm_x + sig_x / 2.) - np.log(cpm_x - sig_x / 2.)) /
             (4 * np.log(C_filt) - 2 * np.log(cpm_x + sig_x / 2.) - 2 * np.log(cpm_x - sig_x / 2.)))

    """
    -----------------  y uncertainty -----------------------
    """

    # get delta C_xy for y direction
    # dC_y = np.zeros(frame_a.shape)  # delta C_i in x direction
    cPlus_y = np.zeros(frame_a.shape)
    cPlus_y[dy:, :] = frame_a[dy:, :] * frame_b_shift[0:-dy, :]
    cMinus_y = np.zeros(frame_a.shape)
    cMinus_y[dy:, :] = frame_a[0:-dy, :] * frame_b_shift[dy:, :]
    dC_y = cPlus_y - cMinus_y
    dC_y = dC_y - np.mean(dC_y)  # remove the mean

    # smooth dC_y in the y direction
    if jit_acceleration:
        dC_y = _smooth_dC(dC_y)
    else:
        for i in range(1, dC_y.shape[0] - 1):
            for j in range(dC_y.shape[1]):
                dC_y[i, j] = (dC_y[i - 1, j] + 2 * dC_y[i, j] + dC_y[i + 1, j]) / 4.

    # calculate covariance matrix
    S_y = np.zeros(frame_a.shape)
    if jit_acceleration:
        S_y = _compute_covariance_matrix_S(frame_a, dC_y, S_y)
    else:
        for i in range(frame_a.shape[0]):
            for j in range(frame_a.shape[1]):

                S0 = dC_y[i, j]
                for k in range(1, 4):
                    try:
                        if dC_y[i + k, j + k] / S0 < 0.05:
                            S_y[i, j] = np.sum(S0 * dC_y[i:i + k, j:j + k])
                            break
                        if (k == 3):
                            S_y[i, j] = np.sum(S0 * dC_y[i:i + k, j:j + k])
                            break
                    except IndexError:
                        # Technically useless, since c is initialised with np.zeros(). Bare except is crap too...
                        S_x[i, j] = 0.

    # smooth and sum the fields
    cPlus_filt_y = (L ** 2) * filt.gaussian_filter(cPlus_y, L)[row_pix, col_pix]
    cMinus_filt_y = (L ** 2) * filt.gaussian_filter(cMinus_y, L)[row_pix, col_pix]
    S_filt_y = (L ** 2) * filt.gaussian_filter(S_y, L)[row_pix, col_pix]

    # calculate standard deviation  of correlation difference
    sig_y = np.sqrt(S_filt_y)
    cpm_y = (cPlus_filt_y + cMinus_filt_y) / 2.

    # final y uncertainty
    Uy_px = ((np.log(cpm_y + sig_y / 2.) - np.log(cpm_y - sig_y / 2.)) /
             (4 * np.log(C_filt) - 2 * np.log(cpm_y + sig_y / 2.) - 2 * np.log(cpm_y - sig_y / 2.)))

    if return_debug_data:
        return Ux_px, Uy_px, (frame_b_shift, u_interp_px, v_interp_px)
    else:
        return Ux_px, Uy_px


if JIT_EXISTS:
    @jit(nopython=True)
    def _bilinear_frame_shift(frame_shift, frame_b, ur, ul, uc, vr, vl, vc):
        # shift second frame
        ny, nx = frame_b.shape
        for i in range(frame_b.shape[0]):
            for j in range(frame_b.shape[1]):
                idx_f00 = i - vl[i, j], j - ul[i, j]
                idx_f01 = i - vc[i, j], j - ul[i, j]
                idx_f10 = i - vl[i, j], j - uc[i, j]
                idx_f11 = i - vc[i, j], j - uc[i, j]
                if idx_f00[0] < ny and idx_f00[1] < nx and idx_f01[0] < ny and idx_f01[1] < nx and idx_f10[0] < ny and \
                        idx_f10[1] < nx and idx_f11[0] < ny and idx_f11[1] < nx:
                    ur_tmp = ur[i, j]
                    vr_tmp = vr[i, j]

                    # get surrounding pixel intensities (fxy)
                    f00 = frame_b[i - vl[i, j], j - ul[i, j]]
                    f01 = frame_b[i - vc[i, j], j - ul[i, j]]
                    f10 = frame_b[i - vl[i, j], j - uc[i, j]]
                    f11 = frame_b[i - vc[i, j], j - uc[i, j]]
                    # do bilinear interpolation
                    frame_shift[i, j] = ((1 - ur_tmp) * (1 - vr_tmp) * f00 + ur_tmp * (1 - vr_tmp) * f10
                                         + (1 - ur_tmp) * vr_tmp * f01 + ur_tmp * vr_tmp * f11)
        return frame_shift
else:
    def _bilinear_frame_shift(frame_shift, frame_b, ur, ul, uc, vr, vl, vc):
        # shift second frame
        ny, nx = frame_b.shape
        for i in range(frame_b.shape[0]):
            for j in range(frame_b.shape[1]):
                idx_f00 = i - vl[i, j], j - ul[i, j]
                idx_f01 = i - vc[i, j], j - ul[i, j]
                idx_f10 = i - vl[i, j], j - uc[i, j]
                idx_f11 = i - vc[i, j], j - uc[i, j]
                if idx_f00[0] < ny and idx_f00[1] < nx and idx_f01[0] < ny and idx_f01[1] < nx and idx_f10[0] < ny and \
                        idx_f10[1] < nx and idx_f11[0] < ny and idx_f11[1] < nx:
                    ur_tmp = ur[i, j]
                    vr_tmp = vr[i, j]

                    # get surrounding pixel intensities (fxy)
                    f00 = frame_b[i - vl[i, j], j - ul[i, j]]
                    f01 = frame_b[i - vc[i, j], j - ul[i, j]]
                    f10 = frame_b[i - vl[i, j], j - uc[i, j]]
                    f11 = frame_b[i - vc[i, j], j - uc[i, j]]
                    # do bilinear interpolation
                    frame_shift[i, j] = ((1 - ur_tmp) * (1 - vr_tmp) * f00 + ur_tmp * (1 - vr_tmp) * f10
                                         + (1 - ur_tmp) * vr_tmp * f01 + ur_tmp * vr_tmp * f11)
        return frame_shift


def image_dewarp(frame_b, x_px, y_px, u_px, v_px, method='bilinear',
                 kx=1, ky=1, jit_acceleration=True):
    """
    Dewarps the second image back onto the first using the
    displacement field and a bilinear sub-pixel interpolation
    scheme.
    Reference: refer to paper 'Analysis of interpolation schemes for image deformation methods in PIV ' 2005

    Parameters
    ----------
    frame_b : `2d array'
        Image array of second snapshot
    x_px, y_px : `2d array`
        y- and y-location of vectors in units of pix!
    u_px, v_px : `2d array`
        x- and y-velocity of vectors in units of pix!
    method : `str`, optional='bilinear'
        type of subpixel image dewarping function.
    kx, ky : ints, optional
        Degrees of the bivariate spline. Default is 1.
    jit_acceleration : `bool`, optional=True
        code speed-up through numba.jit

    Returns
    -------
    frame_b_shift: `2d array`
        2nd frame dewarped back onto the first frame

    """

    # Reverse displacement field to shift frame B back onto frame A. Otherwise it will be moved ahead.
    u_px = -u_px
    v_px = -v_px

    # Interpolate the displacement field onto each pixel
    # using a bilinear interpolation scheme

    # interpolate u and v
    st = time.perf_counter()
    F1 = interp.RectBivariateSpline(y_px[:, 0], x_px[0, :], u_px, kx=kx, ky=ky)
    u_interp_px = F1(range(frame_b.shape[0]), range(frame_b.shape[1]))
    F2 = interp.RectBivariateSpline(y_px[:, 0], x_px[0, :], v_px, kx=kx, ky=ky)
    v_interp_px = F2(range(frame_b.shape[0]), range(frame_b.shape[1]))
    logger.debug(f'interpolation took: {time.perf_counter() - st}')

    # define shifted frame
    frame_shift = np.zeros(frame_b.shape)

    u_interp_px[np.isnan(u_interp_px)] = 0
    v_interp_px[np.isnan(v_interp_px)] = 0

    # get displacement values
    ul = u_interp_px.astype('int32')  # lower int bound of u displacement
    vl = v_interp_px.astype('int32')  # lower int bound of v displacement

    ur = u_interp_px - ul  # remainder of u displacement
    vr = v_interp_px - vl  # remainder of v displacement

    # dewarp the image
    if method == 'bilinear':

        # upper int bound of u and v displacement
        uc = (np.sign(u_interp_px) * np.ceil(np.abs(u_interp_px))).astype('int32')
        vc = (np.sign(v_interp_px) * np.ceil(np.abs(v_interp_px))).astype('int32')

        ur = abs(ur)
        vr = abs(vr)

        # shift second frame
        if jit_acceleration:
            logger.debug('applying jit acceleration for bilinear frame shift')
            st = time.perf_counter()
            frame_shift = _bilinear_frame_shift(frame_shift, frame_b, ur, ul, uc, vr, vl, vc)
            logger.debug(f'bilinear frame shift took: {time.perf_counter() - st}')
        else:
            for i in range(frame_b.shape[0]):
                for j in range(frame_b.shape[1]):
                    try:
                        ur_tmp = ur[i, j]
                        vr_tmp = vr[i, j]

                        # get surrounding pixel intensities (fxy)
                        f00 = frame_b[i - vl[i, j], j - ul[i, j]]
                        f01 = frame_b[i - vc[i, j], j - ul[i, j]]
                        f10 = frame_b[i - vl[i, j], j - uc[i, j]]
                        f11 = frame_b[i - vc[i, j], j - uc[i, j]]
                        # do bilinear interpolation
                        frame_shift[i, j] = ((1 - ur_tmp) * (1 - vr_tmp) * f00 + ur_tmp * (1 - vr_tmp) * f10
                                             + (1 - ur_tmp) * vr_tmp * f01 + ur_tmp * vr_tmp * f11)
                    except IndexError:
                        # index is out of bounds
                        # Set frame to zero
                        frame_shift[i, j] = 0.

        return frame_shift, u_interp_px, v_interp_px

    else:
        # in the future, should add a gaussian method here
        raise ValueError('Image dewarping method not supported. Use bilinear')

# -------------
