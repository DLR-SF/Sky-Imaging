# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
This module provides functionality of merging exposure series of all-sky images for high-dynamic range imaging.
"""

import cv2
import numpy as np

from asi_core.image.hdr.utils import normalize_image, make_weight_lut, remap_intensity_range, tonemap_linear


def correction_oversatured_regions(images, saturation=255):
    """
    Corrects oversaturated regions in a series of images by setting all channels to maximum intensity.

    :param images: List of images as NumPy arrays.
    :param saturation: Saturation threshold (default: 255).
    :return: Tuple of corrected images and a mask indicating non-oversaturated regions.
    """
    # the HDR algorithms will lead to ugly results if one of the channels is saturated (i.e. equal 255), while the others are not.
    # Thus, I check if one of the channels is saturated (or close to it: >= 254) and set all channels to 255.
    all_mask = np.ones_like(images[0])
    for image in images:
        if saturation < 255:
            image = image / saturation * 255
        mask = np.max(image, axis=2) >= 254
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        image[mask] = 255
        all_mask[mask == False] = 0
    return images, all_mask


def merge_exposure_series(
    img_series, 
    exposure_times, 
    algorithm='debevec',
    response=None,
    saturation=255,
    low_clip = 5,
    high_clip = 250,
    weight_type='triangle',
    lnE_range=None,
    apply_tonemapping=True,
    gamma=2.2,
    filetype='.jpg'):
    """
    Merge a series of differently exposed images into a single HDR image.

    :param img_series: Sequence of input images as NumPy arrays (HxWxC, 8-bit or float32).
    :type img_series: list[numpy.ndarray]
    :param exposure_times: Exposure times corresponding to each image (in seconds).
    :type exposure_times: list[float]
    :param algorithm: HDR merging algorithm to use: ``'mertens'`` (exposure fusion),
        ``'debevec'`` (calibrated HDR), or ``'debevec_custom'`` (uses pre-computed response and custom processing).
        Default is ``'debevec'``.
    :type algorithm: str
    :param response: Optional pre-computed camera response function.
    :type response: numpy.ndarray | None
    :param saturation: Pixel value above which intensities are considered saturated.
        Used only for Debevec methods. Default is ``255``.
    :type saturation: int
    :param low_clip: Lower intensity threshold to discard unreliable low values. Default is ``5``.
    :type low_clip: int
    :param high_clip: Upper intensity threshold to discard unreliable high values. Default is ``250``.
    :type high_clip: int
    :param weight_type: Weighting function for Debevec reconstruction, either ``'triangle'`` (default) or ``'sine'``.
    :type weight_type: str
    :param lnE_range: Optional global log-irradiance range ``(min_lnE, max_lnE)`` for consistent scaling.
    :type lnE_range: tuple[float, float] | None
    :param apply_tonemapping: Whether to apply tone mapping to the final HDR image. Default is ``True``.
    :type apply_tonemapping: bool
    :param gamma: Gamma applied to tone-map the HDR result for display. Default is ``2.2``.
    :type gamma: float
    :param filetype: Output file type (``'.jpg'``, ``'.jp2'``, or ``'.png'``). Default ``'.jpg'``.
    :type filetype: str
    :returns: Merged HDR image as a NumPy float32 array.
    :rtype: numpy.ndarray
    """ 
    if (algorithm == 'debevec_custom') and (response is None):
        raise ValueError('You need to pass a pre-computed response to use this algorithm.')

    # To arrays
    img_series = np.asarray(img_series)
    exposure_times = np.array(exposure_times, dtype=np.float32)
    
    # Remap intensities to discard unreliable low/high values
    img_series = remap_intensity_range(img_series, low_clip=low_clip, high_clip=high_clip)

    if algorithm == 'mertens':
        merge_mertens = cv2.createMergeMertens()
        merged_imgs = merge_mertens.process(img_series, exposure_times)  # ,response)
        fmin, fmax = 0, 1.5
    elif algorithm == 'debevec':
        img_series, all_mask = correction_oversatured_regions(img_series, saturation=saturation)
        if response is None:
            calibrate = cv2.createCalibrateDebevec()
            response = calibrate.process(img_series, exposure_times)
        fmin, fmax = 0.00001, 0.03 # heuristics
        merge_debevec = cv2.createMergeDebevec()
        merged_imgs = merge_debevec.process(img_series, exposure_times, response)
        merged_imgs[all_mask == 1] = np.max(merged_imgs)
    elif algorithm == 'debevec_custom':
        merged_imgs = reconstruct_hdr_from_response(
            img_series, exposure_times,
            response=response,
            weight_type=weight_type,
            lnE_range=lnE_range,
        )
        fmin, fmax = 0.0, 1.0
    else:
        raise NotImplementedError(f'Algorithm {algorithm} not implemented')

    # Scale between 0 and 1
    merged_imgs = normalize_image(merged_imgs, min_val=fmin, max_val=fmax)

    # Tone-mapping (e.g., gamma correction), improves translation from hdr to jpg
    if algorithm != "mertens" and apply_tonemapping:
        merged_imgs = tonemap_linear(merged_imgs, gamma=gamma)

    if filetype == '.jpg':
        merged_rescale = (merged_imgs * 255).astype(np.uint8)
    elif filetype == '.jp2' or filetype == '.png':
        merged_rescale = (merged_imgs * 65535).astype(np.uint16)
    else:
        raise ValueError(f'Unsupported file type {filetype}')
    return merged_rescale


def reconstruct_hdr_from_response(
    images,
    exposure_times,
    response,
    weight_type: str = "triangle",
    lnE_range: tuple | None = None,
):
    """
    Debevec-style HDR radiance reconstruction using a precomputed response curve.

    :param images: List or array of HxWxC uint8/float images in 0..255 scale.
    :type images: list[numpy.ndarray] or numpy.ndarray
    :param exposure_times: Array-like of length N containing exposure times for each image.
    :type exposure_times: array-like
    :param response: Per-channel response g(z) in log domain (shape: (256, C)).
    :type response: numpy.ndarray
    :param weight_type: Weighting function for Debevec reconstruction, either ``'triangle'`` (default) or ``'sine'``.
    :type weight_type: str
    :param lnE_range: Optional global log-irradiance range ``(min_lnE, max_lnE)`` for consistent scaling.
    :type lnE_range: tuple[float, float] | None
    :returns: Reconstructed HDR image as a NumPy float32 array in [0, 1] range.
    :rtype: numpy.ndarray
    """
    imgs = np.asarray(images)
    N, H, W, C = imgs.shape
    assert N == len(exposure_times), 'Inconsistent parameters #images={N}, #exposure_times={len(exposure_times)}'

    # Log exposure times
    B = np.log(np.asarray(exposure_times, dtype=np.float32))  # shape (N,)

    # Weigth LUT
    w = make_weight_lut(weight_type) # shape (256,)

    # Normalize response shape to (256, C)
    resp = np.asarray(response, dtype=np.float32).squeeze()
    assert resp.shape == (256, C), f"response must be (256, C), got {resp.shape}"

    hdr = np.zeros((H, W, C), dtype=np.float32)

    i_under = int(np.argmin(exposure_times))
    i_over  = int(np.argmax(exposure_times))

    # Precompute index grid for LUT
    lut_x = np.arange(256, dtype=np.float32)

    for ch in range(C):
        g = resp[:, ch]  # (256,)

        # Z_full: (N, H, W)
        Z_full = imgs[..., ch]

        # Interpolate weights and response at observed Z
        w_interp = np.interp(Z_full, lut_x, w)           # (N,H,W)
        g_interp = np.interp(Z_full, lut_x, g)           # (N,H,W)

        # Weighted solve: lnE = sum_i w(z_ij) * (g(z_ij) - lnΔt_i) / sum_i w(z_ij)
        numerator   = np.sum(w_interp * (g_interp - B[:, None, None]), axis=0)  # (H,W)
        denominator = np.sum(w_interp, axis=0)                                   # (H,W)

        # Identify pixels with no valid weight (all 0)
        mask = denominator <= 1e-12

        # Fallback with under/overexposed sentinel images
        Z_under = imgs[i_under, :, :, ch]
        Z_over  = imgs[i_over,  :, :, ch]
        mask_under = mask & (Z_under <= 0.5)      # ~= 0
        mask_over  = mask & (Z_over  >= 254.5)    # ~= 255

        # Avoid divide-by-zero
        denominator = np.where(denominator == 0.0, 1.0, denominator)

        lnE = numerator / denominator

        # Reasonable lnE bounds from response + exposure span
        max_real = float(np.max(g) - np.min(B))
        min_real = float(np.min(g) - np.max(B))

        lnE[lnE > max_real] = max_real
        lnE[mask_under] = min_real
        lnE[mask_over]  = max_real

        # Map to [0,1] consistently
        if lnE_range is not None:
            gmin, gmax = lnE_range
            ch_lin = normalize_image(np.exp(lnE), min_val=np.exp(gmin), max_val=np.exp(gmax))
        else:
            ch_lin = normalize_image(np.exp(lnE), min_val=np.exp(min_real), max_val=np.exp(max_real))

        hdr[:, :, ch] = ch_lin

    return np.clip(hdr, 0.0, 1.0)
