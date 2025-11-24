# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""Tool for the ASI image data acquisition"""

import os
import pathlib
import pytz
from argparse import ArgumentParser
from datetime import datetime, timedelta, date
import time
import logging

import asi_core.utils.datetime_handling
from asi_core.config import config_loader

from asi_core.real_time import http_image_receiver

# keys expected in camera config file to be able to run the daq
MANDATORY_CFG_KEYS = ['camera_name', 'camera_model', 'latitude', 'longitude', 'altitude', 'daq']

def main():
    """
    Parse cmd line arguments and run data acquisition

    Provide the path to your config file as argument after flag -c or run the daq in a folder which contains a config
    file named 'asi_daq_cfg.yaml'
    """

    parser = ArgumentParser()
    parser.add_argument('-c', '--config', nargs=1, default=[pathlib.Path.cwd() / 'asi_daq_cfg.yaml'])
    timestamp_logfile = datetime.now()

    args = parser.parse_args()
    config_loader.load_config(args.config[0])

    cam_cfg = {'location': {}}
    daq_cfg = []
    for k in MANDATORY_CFG_KEYS:
        temp = config_loader.get(k)
        if k == 'daq':
            cam_cfg['url_cam'] = temp['url_cam']
            storage_path = temp['storage_path']
            daq_working_dir = temp['daq_working_dir']
            if 'process' in temp.keys():
                daq_cfg = temp['process']
        elif k in ['latitude', 'longitude', 'altitude']:
            cam_cfg['location'][k[:3]] = temp
        else:
            cam_cfg[k] = temp

    if type(daq_cfg) is dict:
        if 'sampling_time' in daq_cfg.keys() and daq_cfg['sampling_time'] is not None:
            daq_cfg['sampling_time'] = timedelta(seconds=daq_cfg['sampling_time'])

        if 'night_sampling_time' in daq_cfg.keys() and daq_cfg['night_sampling_time'] is not None:
            daq_cfg['night_sampling_time'] = timedelta(seconds=daq_cfg['night_sampling_time'])

        if 'settling_time' in daq_cfg.keys() and daq_cfg['settling_time'] is not None:
            daq_cfg['settling_time'] = timedelta(seconds=daq_cfg['settling_time'])

        if ('safety_deadtime_between_exposure_series' in daq_cfg.keys() and
                daq_cfg['safety_deadtime_between_exposure_series'] is not None):
            daq_cfg['safety_deadtime_between_exposure_series'] = (
                timedelta(seconds=daq_cfg['safety_deadtime_between_exposure_series']))

    timezone_images = config_loader.get('timezone')
    if timezone_images is not None:
        daq_cfg['timezone_images'] = pytz.timezone(
            asi_core.utils.datetime_handling.timezone_ISO8601_to_pytz_posix(timezone_images))
        timestamp_logfile = timestamp_logfile.astimezone(daq_cfg['timezone_images'])

    storage_path = fr'{storage_path}/{datetime.now():%Y}/{cam_cfg["camera_name"]}'
    os.chdir(daq_working_dir)

    if not os.path.isdir('Logs'):
        os.mkdir('Logs')

    logging.basicConfig(filename=f'Logs/asi_daq_{timestamp_logfile:%Y%m%d_%H%M%S}.log',
                        level=logging.DEBUG, format='%(levelname)s - %(asctime)s: %(message)s', datefmt='%H:%M:%S')

    if cam_cfg['camera_model'] in http_image_receiver.KNOWN_MOBOTIX_MODELS:
        http_image_receiver.MobotixSeriesReceiver(cam_cfg, storage_path, **daq_cfg).record_continuously()
    elif cam_cfg['camera_model'][:4] == 'AXIS':
        http_image_receiver.AxisSeriesReceiver(cam_cfg['url_cam'], storage_path, cam_cfg['location'], **optional_arguments
                           ).record_continuously()
    else:
        raise Exception('Unknown camera type')


if __name__ == '__main__':
    main()
