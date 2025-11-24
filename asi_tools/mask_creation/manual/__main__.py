# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
A small tool to create ASI masks manually.
"""

from pathlib import Path

import argparse

from asi_core.config import config_loader
from asi_tools.mask_creation.manual.mask_creation import ObstacleMaskDetection


if __name__ == '__main__':
    """
    Run the mask creation based on config file
    
    The following is saved:
    - A mat file which contains the mask and the path to the image based on which it was created
    - A jpg image file visualizing the masked areas in the original image
    
    :param -c: Optional path to config file, if not specified, a config file 'mask_creation_cfg.yaml' is expected in 
        the working directory
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs=1, default=[Path(__file__).parent / 'mask_creation_cfg.yaml'])

    args = parser.parse_args()
    config_loader.load_config(args.config[0])
    cfg = config_loader.get('ObstacleMaskDetection')
    
    detector = ObstacleMaskDetection(cfg)
    
    if cfg.get('do_load_existing_mask', False):
        detector.load_existing_mask(cfg['existing_mask_path'])

    detector.refine_manually()

    detector.save_mask_and_docu()
