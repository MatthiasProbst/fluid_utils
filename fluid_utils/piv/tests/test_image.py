import numpy as np

from fluid_utils.piv.image import compute_background
import matplotlib.pyplot as plt

def test_compute_background():
    img2d = np.random.random((10, 10))
    bg = compute_background(img2d, method='random', axis=0)
    assert img2d.shape == bg.shape
    assert np.array_equal(img2d, bg) is False
