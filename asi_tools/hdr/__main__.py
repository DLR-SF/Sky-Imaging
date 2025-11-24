# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
Main tool for high-dynamic range (HDR) generation from exposure series


This script processes a directory of exposure-bracketed images and merges them
into HDR composites. It is intended to be used on a set of images captured
with varying exposure times from all-sky imagers.

The merging is handled by the `asi_core.hdr.process.process_directory` function.
"""

import argparse
from pathlib import Path

from asi_core.image.hdr.pipeline import process_directory
from asi_core.config.logging_config import configure_logging

def parse_arguments():
    """
    Parse command-line arguments for HDR image generation.

    :returns: Parsed arguments including source and destination directories.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Script to merge exposure series to hdr images."
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
        '--save_dir',
        type=str,
        required=True,
        help="Path of target directory to save merged images."
    )
    parser.add_argument(
        '--response_file',
        type=str,
        required=False,
        help="Path of the response file in npz format."
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        required=False,
        default='mertens',
        help="Algorithm name. Choose from 'mertens', 'debevec' and 'debevec_custom'."
    )
    parser.add_argument(
        '--log_file',
        type=str,
        required=False,
        help="Path of log file to write logs."
    )
    return parser.parse_args()


def main():
    """
    Main execution function for HDR merging.

    This function:

    1. Parses command-line arguments.
    2. Converts input paths to `Path` objects.
    3. Calls `process_directory` to merge image series into HDR images.

    :returns: None
    """
    args = parse_arguments()
    log_file = args.log_file
    img_dir = Path(args.image_dir)
    save_dir = Path(args.save_dir)
    response_file = args.response_file
    algorithm = args.algorithm

    configure_logging(log_file=log_file)

    process_directory(img_dir, save_dir, response_file=response_file, algorithm=algorithm)


if __name__ == "__main__":
    main()