# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

import os
import sys
import logging
from functools import wraps

import yaml

try:
    from mmpy_bot import Bot, Settings
    bot_available = True
except ImportError:
    logging.warning('Mattermost bot not available. Skipping mattermost notifications.')
    bot_available = False


def send_mattermost_message(message, config=None):
    if not bot_available:
        return
    if type(config) != dict:
        if config is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mattermost_config.yaml')
        else:
            config_path = config
        try:
            with open(config_path, "r") as config_file:
                config = yaml.load(config_file, yaml.Loader)
        except FileNotFoundError as e:
            logging.warning(f'Could not find config file {config} to setup mattermost notifier.')
            return
    try:
        bot = Bot(
            settings=Settings(
                MATTERMOST_URL=config['mattermost_bot']['url'],
                MATTERMOST_PORT=config['mattermost_bot']['port'],
                MATTERMOST_API_PATH=config['mattermost_bot']['api_path'],
                BOT_TOKEN=config['mattermost_bot']['token'],
                SSL_VERIFY=config['mattermost_bot']['ssl_verify'],
            )
        )
        bot.driver.direct_message(receiver_id=config['mattermost_bot']['receiver_id'], message=message)
    except Exception as e:
        logging.warning(f'Could not send mattermost message. Exception: {e}')


def notify_status(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        script_name = os.path.basename(sys.argv[0])
        try:
            # Execute the main function
            result = func(*args, **kwargs)
            send_mattermost_message(f"✅ {script_name} executed {func.__name__} successfully.")
            return result
        except Exception as e:
            # If an exception occurs, send failure message
            logging.error(f"{func.__name__} in {script_name} failed with error: {str(e)}", exc_info=True)
            send_mattermost_message(f"❌ {script_name} failed in {func.__name__} with error: {str(e)}")
            raise
    return wrapper