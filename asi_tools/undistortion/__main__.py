# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
Undistortion script for all-sky camera images.

This script provides a command-line interface to undistort images taken with 
fisheye or omnidirectional all-sky imagers, based on pre-calibrated camera models.

It supports:
- Configuration via YAML files
- Automated image series processing within a time window
- Application of calibration parameters (internal + external)
- Optional use of binary camera masks to exclude invalid regions
- Saving the undistorted images as JPEG

The internal calibration model follows:
Scaramuzza, D., et al. (2006). A Toolbox for Easily Calibrating Omnidirectional Cameras.
IROS, Beijing, China.

The external orientation (Euler angles) is defined following:
Luhmann, T. (2000). Nahbereichsphotogrammetrie. Wichmann Verlag.
"""

import os
import argparse
from functools import partial
from fastcore.basics import ifnone
from fastcore.parallel import parallel
import yaml
import numpy as np
from pathlib import Path
import logging

from asi_core.camera.sky_imager import AllSkyImager, load_camera_data
from asi_core.utils.filesystem import get_image_files
from asi_core.config.logging_config import configure_logging


logger = logging.getLogger(__name__)
configure_logging()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Undistort all-sky images with config file")
    parser.add_argument('--config', type=str, required=False,
                        help='YAML configuration file including camera and processing parameters')
    parser.add_argument('--image_dir', type=str, required=False,
                        help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=False,
                        help='Directory to save undistorted output images')
    parser.add_argument('--camera_data_file', type=str, required=False,
                        help='YAML file of camera data')
    parser.add_argument('--start_time', type=str, help='(Optional) Start time YYYY-MM-DDTHH:MM:SS')
    parser.add_argument('--end_time', type=str, help='(Optional) End time YYYY-MM-DDTHH:MM:SS')
    parser.add_argument('--resize', type=int, help='(Optional) Resize image to this size')
    parser.add_argument('--apply_camera_mask', type=bool, help='(Optional) Apply camera mask on image')
    parser.add_argument('--limit_angle', type=int, default=78, help='(Optional) Limit zenith angle')
    parser.add_argument('--num_workers', type=int, default=0, help='(Optional) Number of workers')
    return parser.parse_args()


def process_image(image_path, asi=None, image_root=None, output_dir='./undistorted'):
    if image_root is None:
        output_path = Path(output_dir) / image_path.name
    else:
        output_path = Path(output_dir) / Path(image_path)
        image_path = Path(image_root) / Path(image_path)

    image_distorted = asi.load_image(image_path)
    image_undistorted = asi.transform(image_distorted)
    asi.save_image(image_undistorted, output_path)
    return output_path


def main():
    args = parse_arguments()
    config_file = args.config
    if os.path.isfile(config_file):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    if 'Undistortion' in config:
        config = config['Undistortion']

    image_dir = ifnone(args.image_dir, config.get('image_dir'))
    output_dir = ifnone(args.output_dir, config.get('output_dir'))
    camera_data_file = ifnone(args.camera_data_file, config.get('camera_data_file'))
    start_time = ifnone(args.start_time, config.get('start_time'))
    end_time = ifnone(args.end_time, config.get('end_time'))
    resize = ifnone(args.resize, config.get('resize'))
    apply_camera_mask = ifnone(args.apply_camera_mask, config.get('apply_camera_mask'))
    limit_angle = ifnone(args.limit_angle, config.get('limit_angle'))
    num_workers = ifnone(args.num_workers, config.get('num_workers'))

    assert os.path.isdir(image_dir), f'Image directory {image_dir} does not exist'
    assert os.path.isdir(output_dir), f'Output directory {output_dir} does not exist'
    assert os.path.isfile(camera_data_file), f'Camera data config file {camera_data_file} does not exist'

    tfms = {'undistort': True, 'undistort_limit_angle': limit_angle}
    if resize:
        tfms['resize'] = resize
    if apply_camera_mask:
        tfms['apply_camera_mask'] = apply_camera_mask

    camera_cfg = load_camera_data(camera_data_file)
    asi = AllSkyImager(camera_cfg, tfms=tfms)
    if asi.camera_mask is None and apply_camera_mask:
        raise RuntimeError("Camera mask could not be loaded. Aborting to avoid undefined behavior.")

    images = get_image_files(
        config['image_dir'],
        recursive=True,
        as_series=True,
        dt_format="%Y%m%d%H%M%S"
    )
    assert len(images) > 0, f"No images found in {image_dir}"
    if start_time:
        images = images[images.index >= np.datetime64(start_time)]
    if end_time:
        images = images[images.index <= np.datetime64(end_time)]

    relative_image_paths = images.apply(lambda x: x.relative_to(image_dir))
    partial_process_image = partial(process_image, asi=asi, image_root=image_dir, output_dir=output_dir)
    results = parallel(
        partial_process_image,
        relative_image_paths,
        n_workers=num_workers,
        progress=True,
    )

if __name__ == '__main__':
    main()
