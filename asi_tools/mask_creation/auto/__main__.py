# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
Main tool for automatic camera mask creation

This script generates a mask identifying fixed obstructions in sky images,
based on images from a clear day. It computes an average image from a
selection of samples, derives a binary mask, and saves the result along with
a visual overlay.
"""

import pandas as pd
from pathlib import Path
import argparse
from datetime import date

from asi_core.camera.obstacle_mask import create_mask, save_mask 
from asi_core.visualization.masking import visualize_mask
from asi_core.utils.filesystem import get_image_files
from asi_core.image.image_loading import load_image


def parse_arguments():
    """
    Parse command-line arguments for generating a camera mask.

    :returns: Parsed arguments containing image directory, date, camera name,
              output mask directory, and image sampling options.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Script to generate and save a camera mask."
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        required=True,
        help="Path to image directory. Images in subfolders are also considered"
    )
    parser.add_argument(
        '--mask_dir',
        type=str,
        required=False,
        default='./masks',
        help="Path to save calculated mask."
    )
    parser.add_argument(
        '--mask_name',
        type=str,
        required=False,
        default='mask_camera_'+date.today().strftime("%Y%m%d"),
        help="Name of the camera mask file."
    )
    parser.add_argument(
        '--num_images',
        type=int,
        required=False,
        default=40,
        help="Number of images from the image directory used to compute the camera mask."
    )
    parser.add_argument(
        '--image_stride',
        type=int,
        required=False,
        default=100,
        help="Sample every N-th image to span a wider time range (e.g., 100 means use every 100th image)"
    )
    return parser.parse_args()


def main():
    """
    Main entry point for mask creation and visualization.

    This function:

    1. Parses user arguments from the CLI.
    2. Loads images.
    3. Computes an average image and derives a binary mask.
    4. Saves the generated mask and an overlay visualization.

    :raises AssertionError: If the image directory does not exist or contains no images or 
                            the Number of required images is less than the image subset.
    """
    
    args = parse_arguments()
    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    mask_name = args.mask_name
    num_images = args.num_images
    stride = args.image_stride
    
    assert image_dir.is_dir(), f'Image dir {image_dir} does not exist.'
    
    image_files = get_image_files(image_dir)
    assert len(image_files) > 0, f'No images found in {image_dir}'

    img_files_subset = image_files[::stride]
    num_images_subset = len(img_files_subset)
    assert num_images_subset >= num_images, f'Number of required images, {num_images}, \
                                             is less than the actual image subset, {num_images_subset}. \
                                             Consider using different values for num_images, image_stride arguments'

    mask = create_mask(img_files_subset, num_images=num_images)

    if not mask_dir.exists():
        mask_dir.mkdir(parents=True)
    save_mask(mask, mask_dir, mask_name)

    # Visualize results
    an_image = load_image(img_files_subset[int(len(img_files_subset)/2)])
    visualize_mask(an_image, mask, output_file=mask_dir / f'{mask_name}_overlayed_mask.png')


if __name__ == "__main__":
    main()
