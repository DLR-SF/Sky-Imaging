# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
This module provides utility functions of merging exposure series of all-sky images for high-dynamic range imaging.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np
import cv2


def remap_intensity_range(
    image: np.ndarray,
    low_clip: int = 0,
    high_clip: int = 255,
    max_value_range: int = 255,
    dtype: np.dtype = np.uint8,
) -> np.ndarray:
    """
    Linearly remap an image's intensity range [low_clip, high_clip] to [0, max_value_range].

    This function clips all pixel values below low_clip and above high_clip,
    then linearly rescales the remaining range to the specified output range
    (typically [0, 255] for 8-bit images).

    :param image: Input 8-bit array of shape (H, W), (H, W, C), or (N, H, W, C).
    :type image: numpy.ndarray
    :param low_clip: Lower bound (inclusive) of the valid intensity range.
        Values below this threshold are clamped to 0. Default is ``0``.
    :type low_clip: int
    :param high_clip: Upper bound (inclusive) of the valid intensity range.
        Values above this threshold are clamped to max_value_range. Default is ``255``.
    :type high_clip: int
    :param max_value_range: Upper bound of the output domain (255 for 8-bit data).
    :type max_value_range: int
    :param dtype: data type of remapped array. Default is ``np.uint8``.
    :type dtype: np.dtype

    :raises AssertionError: If ``high_clip <= low_clip``.

    :returns: Image remapped to the new intensity range as dtype in [0, max_value_range].
    :rtype: numpy.ndarray

    .. note::
       - This function can be used to remove unreliable tone ends for 8-bit JPEGs
         before calibration or HDR merging.
    """
    assert high_clip > low_clip, 'High clip needs to be higher than low clip'
    img = image.astype(np.float32)

    scale = max_value_range / (high_clip - low_clip)
    out = (np.clip(img, low_clip, high_clip) - low_clip) * scale
    return np.clip(out, 0.0, max_value_range).astype(dtype)


def normalize_image(image, min_val=None, max_val=None):
    """
    Rescales an image to the range [0, 1].

    The function normalizes the input image either using provided minimum and maximum values,
    or computes them from the image itself if not given.

    :param image: Input image as a NumPy array.
    :type image: numpy.ndarray
    :param min_val: Minimum value for rescaling (optional). If None, computed from image.
    :type min_val: float, optional
    :param max_val: Maximum value for rescaling (optional). If None, computed from image.
    :type max_val: float, optional
    :returns: Normalized image with values between 0 and 1.
    :rtype: numpy.ndarray

    :raises ValueError: If `max_val` equals `min_val`, leading to division by zero.
    """
    if min_val is not None and max_val is not None:
        if max_val == min_val:
            raise ValueError("min_val and max_val must be different to avoid division by zero.")
        image = (image - min_val) / (max_val - min_val)
    else:
        min_img = np.min(image)
        max_img = np.max(image)
        if max_img == min_img:
            # Avoid division by zero; return zeros if all values are the same
            image = np.zeros_like(image, dtype=np.float32)
        else:
            image = (image - min_img) / (max_img - min_img)
    return np.clip(image, 0, 1)


def tonemap_linear(
    hdr: np.ndarray,
    method: str = "gamma",
    gamma: float = 2.2,
    exposure: float = 1.0,
    clip: bool = True,
) -> np.ndarray:
    """
    Tone-map a linear HDR image to display-ready [0,1] range.

    :param hdr: Linear HDR image (HxWxC or HxW), float32 or float64.
        Values are expected to be positive; they can exceed 1.0.
    :type hdr: numpy.ndarray
    :param method: Tone-mapping operator to use:
        - ``"gamma"``   : simple power-law gamma correction (default).
        - ``"reinhard"``: photographic tone reproduction.
        - ``"aces"``    : ACES filmic approximation.
    :type method: str
    :param gamma: Gamma value for ``"gamma"`` method. Default is ``2.2``.
    :type gamma: float
    :param exposure: Linear exposure multiplier applied before tone-mapping
        (use to brighten or darken the image globally). Default is ``1.0``.
    :type exposure: float
    :param clip: If True, clamp output to [0,1] range. Default is True.
    :type clip: bool

    :returns: Tone-mapped image in [0,1] as float32.
    :rtype: numpy.ndarray

    .. note::
       This function assumes the input HDR is **linear**.
       If your HDR is already gamma-encoded (e.g. from Mertens fusion),
       skip this step.

    **Examples:**

    .. code-block:: python

        # Basic gamma tone mapping
        ldr = tonemap_linear(hdr, method="gamma", gamma=2.2)

        # Reinhard operator with slight brightening
        ldr = tonemap_linear(hdr, method="reinhard", exposure=1.2)
    """
    hdr = np.asarray(hdr, np.float32)
    hdr = np.maximum(hdr, 0.0) * exposure

    if method == "gamma":
        ldr = np.power(hdr, 1.0 / gamma)
    elif method == "reinhard":
        ldr = hdr / (1.0 + hdr)
    elif method == "aces":
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        ldr = (hdr * (a * hdr + b)) / (hdr * (c * hdr + d) + e)
    else:
        raise ValueError(f"Unknown tone-mapping method: {method!r}")

    if clip:
        ldr = np.clip(ldr, 0.0, 1.0)

    return ldr.astype(np.float32)



def make_weight_lut(weight_type: str = "triangle") -> np.ndarray:
    """
    Return a (256,) weight lookup table in uint8 intensity domain [0..255].

    :param weight_type: Type of weight function to generate.
        Currently supports:
        - ``"triangle"``: Triangular weights centered at 127.5 (default).
        - ``"sine"``: Sine-based weights with cosine bump.
    :type weight_type: str

    :returns: Weight lookup table as a NumPy array of shape (256,) with dtype float32.
    :rtype: numpy.ndarray

    :raises ValueError: If an unsupported ``weight_type`` is specified.
    """
    I = np.arange(256, dtype=np.float32)

    if weight_type == "triangle":
        mid = 127.5
        w = np.where(I <= mid, I, 255.0 - I)
    elif weight_type == "sine":
        # scale to [0,1], then to [-pi, pi], cosine bump to [0,2]
        x = (I / 255.0) * (2.0 * np.pi) - np.pi
        w = np.cos(x) + 1.0
    else:
        raise ValueError(f"Unsupported weight_type: {weight_type}")

    # Ensure non-negative and avoid zeros in denominators later
    w = np.clip(w, 0.0, None)
    return w


def compute_lne_bounds(response_curve: np.ndarray, exposure_times: np.ndarray) -> Tuple[float, float]:
    """
    Compute global lnE (log exposure) range from response curve and exposure times.

    The function calculates the minimum and maximum log exposure values based on the
    camera response curve and the corresponding exposure times.

    :param response_curve: Camera response curve in logarithmic domain. Shape should be
        (256, 1, C) or (256, C), where C is the number of color channels.
    :type response_curve: numpy.ndarray
    :param exposure_times: Array of exposure times corresponding to each image in the series.
        Shape should be (N,), where N is the number of exposures.
    :type exposure_times: numpy.ndarray

    :returns: Tuple of (min_lnE, max_lnE), representing the global log exposure range.
    :rtype: Tuple[float, float]

    .. note::
       This function assumes that the response_curve is in the logarithmic domain (g),
       and exposure_times are in linear units (e.g., seconds).
    """
    g = response_curve[:, 0, :] if (response_curve.ndim == 3 and response_curve.shape[1] == 1) else response_curve
    B = np.log(np.asarray(exposure_times, dtype=np.float32))
    max_realistic = float(np.max(g) - np.min(B))
    min_realistic = float(np.min(g) - np.max(B))
    return min_realistic, max_realistic


