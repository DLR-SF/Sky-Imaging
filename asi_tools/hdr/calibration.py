# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
Main tool for computing calibration response for creating high-dynamic range images from exposure series


This script processes a directory of exposure-bracketed images and merges them
into HDR composites. It is intended to be used on a set of images captured
with varying exposure times from all-sky imagers.

The calibration is handled by the `asi_core.image.hdr.pipeline.calibrate_camera` function.
"""

import argparse
from pathlib import Path

from asi_core.image.hdr.pipeline import calibrate_camera

def parse_arguments():
    """
    Parse command-line arguments for HDR image generation.

    :returns: Parsed arguments including source and destination directories.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Script to calibrate camera for hdr images."
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        required=True,
        help="Path of image source directory, containing exposure series. " \
             "This directory or subfolders of it are expected to contain one image per sampling interval and exposure time. " \
             "Each image file is expected to be named as follows *_<exposure_time>.*" \
             "Where exposure_time is an integer indicating the exposure time (time unit arbitrary)."
    )
    parser.add_argument(
        '--response_file',
        type=str,
        required=True,
        help="Path of target npz file to save response."
    )
    return parser.parse_args()


def main():
    """
    Main execution function for HDR calibration.

    This function:

    1. Parses command-line arguments.
    3. Calls `process_directory` to merge image series into HDR images.

    :returns: None
    """
    args = parse_arguments()

    calibrate_camera(
        image_dir = Path(args.image_dir),
        response_file = Path(args.response_file)
    )


if __name__ == "__main__":
    main()