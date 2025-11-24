# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
This module provides calibration procedure for merging all-sky images for high-dynamic range imaging.
"""
import numpy as np
import random
from typing import Tuple, Optional, List

from asi_core.image.hdr.utils import make_weight_lut
def _make_mesh(sizeX, sizeY):
    """
    Create a regular meshgrid of x and y coordinates.

    :param int sizeX: Number of points along the x‑axis.
    :param int sizeY: Number of points along the y‑axis.
    :returns: Tuple ``(xv, yv)`` where each element is a 2‑D array of shape
              ``(sizeX, sizeY)`` containing the x‑ or y‑coordinates for every
              point in the grid.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    x = np.linspace(0, sizeX - 1, sizeX)
    y = np.linspace(0, sizeY - 1, sizeY)

    xv, yv = np.meshgrid(x, y, indexing="ij")

    return xv, yv


def _make_round_mesh(sizeX, sizeY, max_radius):
    """
    Generate coordinates of points that lie within a circular region of the
    image domain.

    The circular region is centred at the image centre and its radius is
    defined as ``max_radius`` (a fraction of the distance from the centre to the
    closest image border).

    :param int sizeX: Width of the image (number of columns).
    :param int sizeY: Height of the image (number of rows).
    :param float max_radius: Fraction of the smallest half‑dimension that
                             defines the radius of the circular mask. Must be
                             within ``(0, 1]``.
    :returns: Two 1‑D arrays ``(xs, ys)`` containing the x‑ and y‑coordinates of
              the points that fall inside the circular mask.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    xv, yv = _make_mesh(sizeX, sizeY)
    center_x, center_y = sizeX // 2, sizeY // 2

    max_radius = min(center_x, center_y) * max_radius
    radius = np.sqrt((xv - center_x) ** 2 + (yv - center_y) ** 2)
    mask = radius < max_radius
    xs = xv[mask].flatten()
    ys = yv[mask].flatten()

    return xs, ys

def calibrate_response_debevec(samples, exposure_times, smoothing: float = 50.0, weight_type: str = "triangle"):
    """
    Calibrate camera response function using the Debevec method.

    This function estimates the camera response function in logarithmic domain from
    multiple exposures of the same scene. It uses a least squares approach to solve
    for the response function that best fits the measured intensities.

    :param samples:
        List of arrays, each with shape (P_i, N_i, C), where:
        - P_i is the number of pixels in the sample,
        - N_i is the number of exposures for the sample,
        - C is the number of color channels.
    :param exposure_times:
        List of arrays, each with shape (N_i,), containing the exposure times
        corresponding to each sample in ``samples``.
    :param smoothing:
        Smoothing factor used in the smoothness constraint term. Default is 50.0.
    :param weight_type:
        Type of weighting function to use (``'triangle'`` (default) | ``'sine'``).
    :returns:
        Response function in logarithmic domain, shape (256, 1, C).
    """

    C = samples[0].shape[-1]
    n = 256
    response = np.zeros((n, 1, C), dtype=np.float32)
    w_lut = make_weight_lut(weight_type)  # (256,)

    # Precompute total rows for A per channel
    total_data_rows = sum(s.shape[0] * s.shape[1] for s in samples)
    total_irr_rows  = sum(s.shape[0] for s in samples)
    A_rows = total_data_rows + (n - 2) + 1  # data + smoothness + constraint
    A_cols = n + total_irr_rows             # g(0..255) + all logE

    for ch in range(C):
        A = np.zeros((A_rows, A_cols), dtype=np.float32)
        b = np.zeros((A_rows, 1),      dtype=np.float32)

        row = 0
        lE_col_offset = n  # after g entries

        # Data-fitting equations
        for s, t in zip(samples, exposure_times):
            Z = np.clip(s[..., ch].astype(np.int32), 0, 255)  # (P_i, N_i)
            ln_t = np.log(t).astype(np.float32)

            P_i, N_i = Z.shape
            for p in range(P_i):
                for i in range(N_i):
                    z = Z[p, i]                 # intensity
                    wij = w_lut[z]
                    if wij <= 0:
                        continue
                    A[row, z] = wij
                    A[row, lE_col_offset + p] = -wij
                    b[row, 0] = wij * ln_t[i]
                    row += 1
            lE_col_offset += P_i  # advance latent lnE columns

        # Fix g(128) = 0
        A[row, 128] = 1.0
        b[row, 0]   = 0.0
        row += 1

        # Smoothness: for i = 1..n-2: lambda * (g_{i-1} - 2g_i + g_{i+1}) = 0
        lam = float(smoothing)
        for i in range(1, n - 1):
            A[row, i - 1] = lam * w_lut[i]
            A[row, i]     = -2.0 * lam * w_lut[i]
            A[row, i + 1] = lam * w_lut[i]
            row += 1

        # Solve
        x, *_ = np.linalg.lstsq(A[:row], b[:row], rcond=None)
        g = x[:n, 0]  # log response
        response[:, 0, ch] = g

    return response.astype(np.float32)


def get_sample_positions(
    image, 
    samples_per_image=100, 
    max_radius=0.97, 
    sample_technique="random"
):
    """
    Generate sample positions within a circular region of an image.

    This function selects pixel coordinates from within a specified circular area
    of the input image, using either random sampling or histogram-based sampling.

    :param image: Input image array of shape (H, W, C).
    :type image: numpy.ndarray
    :param samples_per_image: Number of sample positions to generate. Default is 100.
    :type samples_per_image: int
    :param max_radius: Fraction of the smaller dimension to define the maximum radius. Default is 0.97.
    :type max_radius: float
    :param sample_technique: Sampling method, either 'random' or 'histogram'. Default is 'random'.
    :type sample_technique: str
    :returns: Tuple of arrays containing x and y coordinates of sampled points.
    :rtype: Tuple[numpy.ndarray, numpy.ndarray]
    """
    sizeX, sizeY = image.shape
    xs, ys = _make_round_mesh(sizeX, sizeY)
    if sample_technique == "random":
        random_subset = random.sample(range(len(xs)), samples_per_image)
        xs = xs[random_subset].astype(int)
        ys = ys[random_subset].astype(int)
    elif sample_technique == "histogram":
        xs_original = xs
        ys_original = ys
        xs = []
        ys = []
        bins = 10
        I_min = 0
        I_max = 255 - 1
        Is = np.mean(image, axis=2)
        Is = Is[mask].flatten()
        limits = np.linspace(I_min, I_max, bins + 1)
        for i in range(bins):
            mask_bin = np.logical_and(limits[i] <= Is, Is <= limits[i + 1])
            random_subset = random.sample(
                range(np.sum(mask_bin)),
                min(np.sum(mask_bin), samples_per_image // bins),
            )
            xs.extend(xs_original[mask_bin][random_subset])
            ys.extend(ys_original[mask_bin][random_subset])
        xs = np.array(xs).astype(int)
        ys = np.array(ys).astype(int)
    else:
        raise ValueError(f"Sampling technique {sample_technique} is not implemented.")
    return xs, ys
