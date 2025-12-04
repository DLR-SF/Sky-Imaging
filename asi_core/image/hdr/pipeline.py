# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
This module provides all processing steps for merging exposure series of all-sky images for high-dynamic range imaging.
"""

import os
import pandas as pd
import cv2
import re
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
import random
import json
import warnings
from tqdm import tqdm
from fastcore.parallel import parallel
from typing import Tuple, Optional, List, Dict, Any

from asi_core.image.image_loading import load_image, load_images
from asi_core.utils.datetime_handling import parse_datetime
from asi_core.utils.filesystem import get_image_files
from asi_core.image.hdr.merge import merge_exposure_series
from asi_core.image.hdr.calibration import get_sample_positions, calibrate_response_debevec
from asi_core.image.hdr.utils import compute_lne_bounds, remap_intensity_range


_EXPO_RE = re.compile(r"_([0-9]+)\.(jpg|jpeg|png|jp2)$", re.IGNORECASE)


def _parse_exposure_from_name(name: str) -> Optional[int]:
    """
    Parse exposure time (integer) from filename tail: \*_<EXPO>.<ext>
    """
    m = _EXPO_RE.search(name)
    return int(m.group(1)) if m else None


def _group_files_by_time(directory: str | Path, round_ts_to: str = "30s") -> pd.core.groupby.DataFrameGroupBy:
    """
    Group image files found in `directory` by rounded timestamp (e.g., every 30s) into exposure brackets.
    Returns a pandas groupby indexed by the rounded timestamp.
    """
    files = get_image_files(directory)
    if not files:
        df_empty = pd.DataFrame({"timestamp_rounded": [], "filepath": []})
        return df_empty.groupby("timestamp_rounded")

    df = pd.DataFrame({"filepath": files})
    df["timestamp"] = df["filepath"].map(lambda p: parse_datetime(Path(p).name))
    df = df.dropna(subset=["timestamp"])
    df["timestamp_rounded"] = df["timestamp"].dt.floor(round_ts_to)
    return df.groupby("timestamp_rounded")

def _save_response_plot(
    response: np.ndarray,
    out_path: str | Path,
    include_mean: bool = True,
    dpi: int = 150,
) -> Path:
    """
    Plot the camera log-response g(z) per channel and save the figure.

    :param response: Response curve in LOG domain. Shape (256, 1, C) or (256, C).
    :type response: np.ndarray
    :param out_path: Output image path (e.g., 'response_curve.png').
    :type out_path: str | Path
    :param include_mean: If True, also plot the channel-mean curve. Default is `True`.
    :type include_mean: bool
    :param dpi: Figure DPI. Default is `150`.
    :type dpi: int
    :returns: The saved file path.
    :rtype: Path
    """
    import matplotlib.pyplot as plt
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    g = np.asarray(response, dtype=np.float32)
    if g.ndim == 3 and g.shape[1] == 1:
        g = g[:, 0, :]             # (256, C)
    if g.ndim == 1:
        g = g[:, None]             # (256, 1)

    z = np.arange(g.shape[0])
    C = g.shape[1]

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=dpi)
    ax.set_title("Camera Log-Response g(z)")
    ax.set_xlabel("Intensity z (0–255)")
    ax.set_ylabel("g(z)  (log exposure)")

    # Plot each channel; let Matplotlib choose default colors
    for c in range(C):
        ax.plot(z, g[:, c], label=f"ch {c}")

    if include_mean and C > 1:
        g_mean = np.mean(g, axis=1)
        ax.plot(z, g_mean, linestyle="--", linewidth=1.25, label="mean")

    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def save_response_curve(
    path: str | Path,
    response: np.ndarray,
    metadata: dict = None,
    save_plot: bool = True,
    plot_filename: str = 'response_curve.jpg'
    ) -> None:
    """
    Save response curve to npz file in LOG domain (Debevec g(z)) with JSON-formatted metadata.

    The response curve is stored in a .npz file with the key 'response' containing an array of shape (256, C),
    where C is the number of color channels. Metadata is saved as a JSON string under the key 'metadata'.

    If ``save_plot`` is True, a plot of the response curve is also saved as a JPEG file in the same directory.

    :param path: The output file path for the .npz file.
    :type path: str | Path
    :param response: The response curve data of shape (256, C) or (256, 1, C).
    :type response: np.ndarray
    :param metadata: A dictionary containing metadata to be saved with the curve.
    :type metadata: dict, optional
    :param save_plot: Whether to save a plot of the response curve.
    :type save_plot: bool
    :param plot_filename: The filename of the plot of the response curve.
    :type plot_filename: str
    :returns: None
    :rtype: None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = json.dumps(metadata or {}, ensure_ascii=False)
    np.savez(path, response=response.astype(np.float32), metadata=np.array(meta))
    if save_plot:
        plot_path = path.parent / plot_filename
        _save_response_plot(response=response, out_path=plot_path)


def load_response_curve(path: str | Path) -> np.ndarray:
    """
    Load response curve from npz, returning (256, C) in LOG domain and metadata dict.
    """
    with np.load(path, allow_pickle=False) as f:
        response = f["response"]               # (256,1,C) or (256,C)
        meta = json.loads(str(f["metadata"]))  # dict
    return response.astype(np.float32), meta


def calibrate_camera(
    image_dir: str | Path,
    response_file: str | Path,
    samples_per_image: int = 1000,
    sample_technique: str = "random",
    max_processed_groups: int = 10,
    smoothness: float = 50.0,
    weight_type: str = "triangle",
    low_clip: int = 5,
    high_clip: int = 250,
    seed: Optional[int] = None,
    round_ts_to: str = "30s",
) -> np.ndarray:
    """
    Calibrate camera response (Debevec) from a subset of exposure brackets in a directory.

    - Groups files by rounded timestamp (e.g., every 30s).
    - For up to ``max_processed_groups`` groups:
        * parses and sorts images by exposure time,
        * samples P pixels (xs, ys),
        * builds stacks (P, N, C) across N exposures,
        * (optionally) clips dynamic range for stability,
        * calibrates response g(z) per channel.
    - Saves response to ``response_file`` (NPZ of shape (256, C), LOG domain).
    - Returns response as (256, 1, C) in LOG domain.

    Notes:
    - Assumes filenames like: YYYYmmddHHMMSS_<EXPO>.<ext>, where <EXPO> is an integer.

    :param image_dir: Directory containing image files grouped by timestamp.
    :type image_dir: str | Path
    :param response_file: Output file path for saving the calibrated response curve.
    :type response_file: str | Path
    :param samples_per_image: Number of pixel samples to extract from each image. 
        Default is ``1000``.
    :type samples_per_image: int
    :param sample_technique: Sampling method (``'random'``` (default) | ``'histogram'```).
    :type sample_technique: str
    :param max_processed_groups: Maximum number of timestamp groups to process. Default is ``10``.
    :type max_processed_groups: int
    :param smoothness: Smoothing factor for the response curve fitting. Default is ``50.0``.
    :type smoothness: float
    :param weight_type: Weighting function type for calibration 
        ((``'triangle'``` (default) | ``'sine'```).
    :type weight_type: str
    :param low_clip: Lower intensity threshold to discard unreliable low values. Default is ``5``.
    :type low_clip: int
    :param high_clip: Upper intensity threshold to discard unreliable high values. Default is ``250``.
    :type high_clip: int
    :param seed: Optional random seed for reproducibility.
    :type seed: Optional[int]
    :param round_ts_to: Time resolution for grouping images (e.g., ``'30s'`` (default), ``'1min'``).
    :type round_ts_to: str
    :returns: Camera response curve in LOG domain with shape (256, 1, C).
    :rtype: np.ndarray
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    groups = _group_files_by_time(image_dir, round_ts_to=round_ts_to)

    keys = list(groups.groups.keys())
    if not keys:
        raise ValueError(f"No calibratable image groups found in {image_dir}")

    # Randomly pick a subset of groups
    pick = random.sample(keys, k=min(len(keys), max_processed_groups))

    samples_list: List[np.ndarray] = []
    expt_sets: List[np.ndarray] = []

    for key in pick:
        group = groups.get_group(key)
        files = group["filepath"].tolist()

        # Parse exposures and keep only valid ones
        parsed = [(fp, _parse_exposure_from_name(Path(fp).name)) for fp in files]
        parsed = [(fp, expo) for fp, expo in parsed if expo is not None]
        if len(parsed) < 2:
            continue

        # Sort by exposure time (ascending)
        parsed.sort(key=lambda t: t[1])
        files_sorted = [fp for fp, _ in parsed]
        expt_sorted = np.array([expo for _, expo in parsed], dtype=np.float32)

        # Sample positions from the mid-exposure image
        mid_file = files_sorted[len(files_sorted) // 2]
        img_mid = load_image(mid_file)
        img_mid = remap_intensity_range(img_mid, low_clip=low_clip, high_clip=high_clip)

        xs, ys = get_sample_positions(
            img_mid,
            samples_per_image=samples_per_image,
            sample_technique=sample_technique,
        )

        P = len(xs)
        N = len(files_sorted)
        C = img_mid.shape[-1]
        stack = np.zeros((P, N, C), dtype=np.float32)

        for i, fp in enumerate(files_sorted):
            im = load_image(fp)
            stack[:, i, :] = im[xs, ys, :]

        # maps camera's useful range into [0,255]
        stack = np.clip(((stack.astype(np.float32) - 5) / 235 * 255), 0, 255)

        samples_list.append(stack)
        expt_sets.append(expt_sorted)

    if not samples_list:
        raise ValueError(f"Could not assemble any valid calibration stacks in {image_dir}")

    # Calibrate
    response = calibrate_response_debevec(
        samples=samples_list,
        exposure_times=expt_sets,
        smoothing=smoothness,
        weight_type=weight_type,
    )

    # Compute lnE bounds for metadata
    all_exposures = np.concatenate(expt_sets)
    min_lnE, max_lnE = compute_lne_bounds(response, all_exposures)

    # Build metadata
    metadata: Dict[str, Any] = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "input_directory": str(image_dir),
        "samples_per_image": samples_per_image,
        "sample_technique": sample_technique,
        "max_processed_groups": max_processed_groups,
        "smoothness": smoothness,
        "weight_type": weight_type,
        "low_clip": low_clip,
        "high_clip": high_clip,
        "seed": seed,
        "num_groups_used": len(samples_list),
        "channels": int(response.shape[-1]),
        "lnE_range": [float(min_lnE), float(max_lnE)],
        "file_format": "log-response",
        "algorithm": "Debevec"
    }

    # Save response + metadata
    save_path = Path(response_file)
    save_response_curve(save_path, response, metadata)
    print(f"Saved response curve to {save_path}")
    print(f"lnE range: {min_lnE:.3f} – {max_lnE:.3f}")

    return response, metadata


def get_lne_range(
    image_dir: str | Path,
    response_file: str | Path,
    round_ts_to: str = "30s",
) -> Tuple[float, float]:
    """
    Compute a global (min_lnE, max_lnE) using the saved NPZ response curve and all
    exposure times present in `directory`. Useful for consistent normalization.

    :param image_dir: Directory containing image files grouped by timestamp.
    :type image_dir: str | Path
    :param response_file: Path to the saved NPZ response curve file.
    :type response_file: str | Path
    :param round_ts_to: Time resolution for grouping images (e.g., '30s' (default), '1min').
    :type round_ts_to: str
    :returns: A tuple of (min_lnE, max_lnE) representing the global log exposure range.
    :rtype: Tuple[float, float]
    """
    # Load calibrated response (log domain), shape normalized to (256, 1, C)
    response_curve, _meta = load_response_curve(Path(response_file))

    # Collect exposure times from all grouped files
    groups = _group_files_by_time(image_dir, round_ts_to=round_ts_to)
    exposures: List[int] = []
    for _, group in groups:
        for fp in group["filepath"]:
            expo = _parse_exposure_from_name(Path(fp).name)
            if expo is not None:
                exposures.append(expo)

    if not exposures:
        raise ValueError(f"No exposure times parsed in {image_dir}")

    exposures = np.asarray(exposures, dtype=np.float32)
    return compute_lne_bounds(response_curve, exposures)


def create_and_save_hdr(img_series, exposure_times, output_path, **kwargs_merging):
    """
    Creates an HDR image from a series of exposures and saves it to a file.

    :param img_series: List of images as NumPy arrays.
    :param exposure_times: List of exposure times corresponding to the images.
    :param output_path: Path where the HDR image will be saved.
    :param kwargs_merging: Additional parameters for the merging function.
    """
    filetype = Path(output_path).suffix
    merged_img = merge_exposure_series(img_series=img_series, exposure_times=exposure_times, filetype=filetype,
                                       **kwargs_merging)
    Path(output_path).parent.mkdir(exist_ok=True)
    if filetype == 'png':
        cv2.imwrite(str(output_path), merged_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    else:
        cv2.imwrite(str(output_path), merged_img)
    logging.info(f"Saved HDR image to {output_path}")


def process_timestamp(timestamp_group, root_dir, target_dir):
    """Process all images corresponding to a single timestamp."""
    try:
        timestamp, image_paths = timestamp_group
        exposure_times = image_paths.index.get_level_values(1)
        if len(image_paths) < 3: return False
        # Ensure images are sorted properly
        images = load_images(image_paths.sort_index().apply(lambda x: Path(root_dir) / Path(x)), format='bgr')
        # Construct output path based on relative image paths
        relative_path = Path(image_paths.iloc[0]).parent
        output_path = target_dir / relative_path / timestamp.strftime('%Y%m%d%H%M%S_hdr.jpg')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        create_and_save_hdr(images, exposure_times, output_path)
    except:
        return False
    return True


def process_hdr_series(asi_files: pd.Series, root_dir: str, target_dir: str, n_workers=0):
    """
    Process a multi-index Pandas Series containing relative image paths to generate HDR images.

    :param asi_files: A multi-index Pandas Series where the index consists of timestamps and exposure times, and the values contain relative image paths.
    :type asi_files: pd.Series
    :param root_dir: The root directory containing all the source images.
    :type root_dir: str
    :param target_dir: The target directory where the generated HDR images will be stored (relative paths with respect to root_dir are retained).
    :type target_dir: str
    :param n_workers: The number of parallel workers to use for processing. Defaults to 0 (no parallelism).
    :type n_workers: int, optional
    :return: A Pandas Series with timestamps as the index and generated HDR file paths as values.
    :rtype: pd.Series
    """
    target_dir = Path(target_dir)

    # Group images by the primary timestamp index and process in parallel
    timestamps = asi_files.index.get_level_values(0).unique()
    results = parallel(process_timestamp, asi_files.groupby(level=0), root_dir=root_dir, target_dir=target_dir,
                       n_workers=n_workers, total=len(timestamps), progress=True)
    return pd.Series(results, index=timestamps, name='created_hdr')


def process_directory(directory, save_dir, round_ts_to='30s', response_file=None, **kwargs):
    """
    Processes a directory of images by grouping them into short time intervals and creating HDR images.

    This function performs the following steps:

    1. Groups image files in the specified directory by timestamp with a given rounding interval.
    2. If a response file is provided, it loads the camera response curve.
    3. For each group of images with similar timestamps, it creates an HDR image.
    4. Saves the resulting HDR images to the specified output directory.

    :param directory: Path to the directory containing images.
    :type directory: str
    :param save_dir: Path to the directory where HDR images will be saved.
    :type save_dir: str
    :param round_ts_to: Time resolution for grouping images (e.g., '30s' (default), '1min').
    :type round_ts_to: str
    :param response_file: Path to the saved NPZ response curve file (optional).
    :type response_file: str, optional
    :param kwargs: Additional keyword arguments passed to the HDR merging function.
    :type kwargs: dict
    :returns: None
    :rtype: None
    """
    directory = Path(directory)
    save_dir = Path(save_dir)
    assert directory.exists(), f'Directory {directory.absolute()} does not exist.'

    if response_file is not None:
        response, meta_data = load_response_curve(response_file)
        logging.info(f"Loaded response from {Path(response_file).absolute()}")
        if "lnE_range" in meta_data:
            lnE_range = float(meta_data["lnE_range"][0]), float(meta_data["lnE_range"][1])
        else:
            lnE_range = None
    else:
        response = None

    logging.info(f"Process directory {directory.absolute()} and save results in {save_dir.absolute()}")
    unprocessed_files = get_image_files(directory, as_series=True)
    logging.info(f"Found {len(unprocessed_files)} images in total.")

    df = unprocessed_files.rename("filepath").reset_index().rename(columns={'index': 'timestamp'})
    if round_ts_to is not None:
        df['timestamp_rounded'] = pd.to_datetime(df['timestamp']).dt.floor(round_ts_to)
    else:
        df['timestamp_rounded'] = pd.to_datetime(df['timestamp'])
    df['filename'] = df['filepath'].apply(lambda p: Path(p).name)
    df['exposure_time'] = df['filename'].str.extract(r'_([0-9]+)\.jpg$')[0].astype(int)
    grouped = df.groupby('timestamp_rounded')
    logging.info(f"Creating {len(grouped)} hdr images.")

    for timestamp, group in tqdm(grouped):
        images = []
        exposure_times = []
        rel_path = group.iloc[0]['filepath'].relative_to(directory)
        try:
            with warnings.catch_warnings(record=True) as warn_list:
                for i, row in group.iterrows():
                    current_file = row['filepath']
                    current_exposure = row['exposure_time']
                    images.append(load_image(current_file, format='bgr'))
                    exposure_times.append(current_exposure)
                new_image_name = timestamp.strftime("%Y%m%d%H%M%S_hdr.jpg")
                new_image_file = save_dir / rel_path.parent / new_image_name
                create_and_save_hdr(images, exposure_times, new_image_file, response=response, lnE_range=lnE_range, **kwargs)
                if warn_list:
                    for w in warn_list:
                        logging.warning(f"{new_image_file} created with warning {w.message}")
        except Exception as e:
            logging.exception(f"Iteration for timestamp {timestamp} failed with error {e}")
