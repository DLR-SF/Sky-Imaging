# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

import re
from datetime import datetime

import pandas as pd
import pytz
from scipy import stats


def get_ETC_GMT_timezone(desired_timezone="GMT"):
    """
    Convert the given Etc/GMT timezone into a `pytz` timezone object.

    Example: UTC+2: get_ETC_GMT_timezone('GMT+2') returns pytz timezone object for "Etc/GMT-2"

    :param desired_timezone: (str) A string specifying the desired GMT timezone. It should be in the format 'GMT+1'
        (in the case of UTC+1). If a '+' sign is used, it will be replaced with '-' to follow the Etc/GMT convention
        (https://en.wikipedia.org/wiki/Tz_database#Area).
    :return: (pytz.timezone), Etc/GMT timezone
    """

    if '+' in desired_timezone:
        desired_timezone = desired_timezone.replace('+', '-')

    elif '-' in desired_timezone:
        desired_timezone = desired_timezone.replace('-', '+')

    return pytz.timezone('Etc/' + desired_timezone)


def parse_datetime(dt_string, datetime_format="%Y%m%d%H%M%S", raise_on_fail=False):
    """
    Extract timestamp from string into datetime object.

    :param dt_string: Timestamp string
    :type dt_string: str
    :param datetime_format: Datetime format template with the following placeholders:
        %Y year, %m month, %d day, %H hour, %M minute, %S second. Default is "%Y%m%d%H%M%S".
    :type datetime_format: str                  
    :param raise_on_fail: Flag to decide whether to raise an error or return None
        if no timestamp could be extracted. Default is ``False``.
    :type raise_on_fail: bool
    :return: Timestamp from given string.
    :rtype: datetime.datetime
    """

    pattern = datetime_format.replace('%Y', r'\d{4}')
    pattern = pattern.replace('%y', r'\d{2}')
    pattern = pattern.replace('%m', r'\d{2}')
    pattern = pattern.replace('%d', r'\d{2}')
    pattern = pattern.replace('%H', r'\d{2}')
    pattern = pattern.replace('%M', r'\d{2}')
    pattern = pattern.replace('%S', r'\d{2}')
    match = re.search(pattern, dt_string)

    if match is None:
        if raise_on_fail:
            raise ValueError(f"Could not extract timestamp from filename: {dt_string}")
        else:
            timestamp = None
    else:
        timestamp_str = match.group(0)
        timestamp = datetime.strptime(timestamp_str, datetime_format)

    return timestamp


def get_temporal_resolution_from_timeseries(data):
    """
    Get temporal resolution from a pandas DataFrame with a DatetimeIndex.
    Resolution is rounded to seconds

    :param data: (DataFrame) DataFrame with datetime index
    :return: (Timedelta) DataFrame time resolution
    """

    return pd.Timedelta(stats.mode(data.index.to_series().diff().dt.round('1s'), keepdims=False).mode)


def timezone_ISO8601_to_pytz_posix(tz):
    """
    Convert from ISO 8601 to posix convention (ISO 8601: Positive east of Greenwich)

    :param tz: (str) A string specifying the desired GMT timezone. It should be in the format 'GMT+1'
        (in the case of UTC+1). If a '+' sign is used, it will be replaced with '-' to follow the Etc/GMT convention
        (https://en.wikipedia.org/wiki/Tz_database#Area).
    :return: (str), Etc/GMT timezone
    """

    if 'GMT' in tz:
        if '+' in tz:
            tz = tz.replace('+', '-')
        elif '-' in tz:
            tz = tz.replace('-', '+')
        tz = 'Etc/' + tz

    return tz
