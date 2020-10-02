import numpy as np
import scipy.ndimage as ndimage

def circular_gradient(x, step_size=1.0, periodicity=None):
    if periodicity is None:
        periodicity = 2 * np.pi

    g = np.exp(1j * x * 2 * np.pi / periodicity)

    ds = step_size / 2
    gx1 = np.fft.ifft2(ndimage.fourier_shift(np.fft.fft2(g), shift=(0, ds)))
    gx2 = np.fft.ifft2(ndimage.fourier_shift(np.fft.fft2(g), shift=(0, -ds)))

    argsx = (np.angle(gx1 * np.conj(gx2)) / step_size) * periodicity / np.pi / 2

    gy1 = np.fft.ifft2(ndimage.fourier_shift(np.fft.fft2(g), shift=(ds, 0.)))
    gy2 = np.fft.ifft2(ndimage.fourier_shift(np.fft.fft2(g), shift=(-ds, 0.)))
    argsy = (np.angle(gy1 * np.conj(gy2)) / step_size) * periodicity / np.pi / 2
    return np.array([argsx, argsy])