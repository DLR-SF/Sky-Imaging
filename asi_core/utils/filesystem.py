# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

import logging
import os
import re
import shutil
from pathlib import Path

import pandas as pd
from fastcore.parallel import parallel

from asi_core.utils.datetime_handling import parse_datetime
from asi_core.config.constants import IMAGE_EXTENSIONS, IGNORED_PATTERNS_SUBDAY, FSTRING_RE


def get_absolute_path(filepath, root=None, as_string=False):
    """
    Make filepath an absolute path. It can be combined with a root path.

    :param filepath: (str) file path.
    :param root: (str) root path. Default is None.
    :param as_string: (bool) if True, return absolute path as a string. Default is False.
    :return: (Path or str) absolute path.
    """

    absolute_path = Path(filepath)

    if root is not None:
        absolute_path = Path(root) / filepath

    absolute_path = absolute_path.resolve()

    if as_string:
        absolute_path = str(absolute_path)

    return absolute_path


def replace_double_backslashes_with_slashes_in_path(str_path, root_dir=None):
    """
    Replace double backslashes from windows paths with slashes.

    :param str_path: (str) file path
    :param root_dir: (str) root path. Default is None.
    :return: (Path) String path as a Path object.
    """

    str_path_with_slashes = Path(str_path.replace('\\', '/'))

    if root_dir is not None:
        str_path_with_slashes = Path(root_dir.replace('\\', '/')) / str_path_with_slashes

    return str_path_with_slashes

def _get_files(p, fs, extensions=None, substring=None):
    """
    Get all files in path with 'extensions' and a name containing a 'substring'.

    :param p: (str) directory path
    :param fs: (list str) filenames.
    :param extensions: (str) File extension to filter file list. Default is None.
    :param substring: (str) Substring in filename to filter file list. Default is None.
    :return: (Path) String path as a Path object.
    """

    p = Path(p)
    res = [p / f for f in fs if not f.startswith('.')
           and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)
           and ((not substring) or substring in f)]

    return res


def get_files(path, extensions=[], substring=None, recursive=True, folders=[], followlinks=True):
    """
    Get all files in `path` with optional `extensions` or `substring`, optionally `recursive`, only in `folders`,
    if specified.

    :param path: (str) directory path
    :param extensions: (list str) File extensions to filter file list, i.e. [".txt", ".jpg"]. Default is [].
    :param substring: (str) Substring in filename to filter file list. Default is None.
    :param recursive: (bool) If True, visit files in subfolders. Default is True.
    :param folders: (list str) Subfolders to visit. Default is [].
    :param followlinks: (bool) If True, visit directories pointed to by symlinks, on systems that support them.
        Default is True.
    :return: (list Path) File paths.
    """

    path = Path(path)

    extensions = {e.lower() for e in extensions}

    if recursive:
        res = []
        for i, (p, d, f) in enumerate(os.walk(path, followlinks=followlinks)):
            d.sort()
            f.sort()
            if len(folders) != 0 and i == 0:
                d[:] = [o for o in d if o in folders]
            else:
                d[:] = [o for o in d if not o.startswith('.')]
            if len(folders) != 0 and i == 0 and '.' not in folders:
                continue
            res += _get_files(p, f, extensions, substring)
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        res = _get_files(path, f, extensions, substring)
        res.sort(key=lambda p: str(p))
    return res


def get_image_files(path, recursive=True, folders=[], extensions=IMAGE_EXTENSIONS, substring=None, as_series=False,
                    dt_format="%Y%m%d%H%M%S", round_to=None):
    """
    Get image files in `path` recursively, only in `folders` and with `substring`, if specified.

    :param path: (str) directory path
    :param recursive: (bool) If True, visit files in subfolders. Default is True.
    :param folders: (list str) Subfolders to visit. Default is [].
    :param extensions: (list str) File extension to filter file list. Default is IMAGE_EXTENSIONS.
    :param substring: (str) Substring in filename to filter file list. Default is None.
    :param as_series: (bool) if True, return a pandas series. Default is False.
    :param dt_format: (str) Datetime format template with the following placeholders:

        - %Y year,
        - %m month,
        - %d day,
        - %H hour,
        - %M minute,
        - %S second.
    Default is "%Y%m%d%H%M%S".
    :param round_to: (str) Value to round to, i.e. '1D', '2H', '30s'. When None no rounding is done. Default is None.
    :return: (list Path or Pandas series) Image file paths.
    """

    img_files = get_files(path, extensions=extensions, substring=substring, recursive=recursive, folders=folders)

    if as_series:
        img_files = image_filelist_to_pandas_series(img_files, dt_format=dt_format, round_to=round_to)

    return img_files


def image_filelist_to_pandas_series(img_files, dt_format="%Y%m%d%H%M%S", round_to=None, drop_nat=True):
    """
    Convert list of image files with datetime names into pandas Series.

    :param img_files: (list Path) Image file paths
    :param dt_format: (str) Datetime format template with the following placeholders:
                            %Y year, %m month, %d day, %H hour, %M minute, %S second.
                            Default is "%Y%m%d%H%M%S".
    :param round_to: (str) Value to round to, i.e. '1D', '2H', '30s'. When None no rounding is done. Default is None.
    :param drop_nat: (bool) Flag to determine whether to drop entries where the timestamp could not be extracted.
        Default is True.
    :return: (Pandas series) Image file paths.
    """

    series = pd.Series(img_files)
    series.index = series.apply(lambda x: parse_datetime(Path(x).stem, datetime_format=dt_format))

    if drop_nat:
        series = series[series.index.notnull()]

    if round_to is not None:
        series.index = series.index.round(round_to)

    return series


def copy_file(source_file, target_file=None, target_directory=None, create_parents=False):
    """
    Copy a file to a specified target file or directory.

    :param source_file: (str) Path to the source file.
    :param target_file: (str) Path to the target file (optional if target_directory is provided).
    :param target_directory: (str) Path to the target directory (optional if tgt_file is provided).
    :param create_parents: (bool) If True parent directories are created if don't exist. Default is False.
    """

    assert not (target_file is None and target_directory is None), 'Either target_file or target_directory must be specified.'

    if target_file is None:
        target_file = Path(target_directory) / Path(source_file).name

    if create_parents:
        Path(target_file).parent.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy(str(source_file), str(target_file))
        logging.info(f"Successfully copied {source_file} to {target_file}")
    except FileNotFoundError:
        logging.error(f"Error: File {source_file} not found.")
    except IsADirectoryError:
        logging.error(f"Error: Destination {target_file} is a directory, not a file.")
    except Exception as e:
        logging.info(f"Error: {e}")


def copy_file_relative_to_directories(relative_filepath, source_directory, target_directory):
    """
    Copy a file from the source directory to the target directory while preserving its relative path.

    :param relative_filepath: (str) File path relative to the source directory.
    :param source_directory: (str) Root source directory.
    :param target_directory: (str) Root target directory.
    """
    absolute_source_filepath = get_absolute_path(relative_filepath, root=source_directory)
    absolute_target_filepath = get_absolute_path(relative_filepath, root=target_directory)

    Path(absolute_target_filepath).parent.mkdir(parents=True, exist_ok=True)

    copy_file(absolute_source_filepath, absolute_target_filepath)


def parallel_copy_files(filepaths, source_directory, target_directory, keep_dir_structure=False, num_workers=0):
    """
    Copy multiple files in parallel, optionally preserving directory structure.

    :param filepaths: (List str) File paths to copy.
    :param source_directory: (str) Source directory (used to determine relative paths).
    :param target_directory: (str) Target directory where files will be copied.
    :param keep_dir_structure: (bool) If True maintain the original directory structure. Default is False.
    :param num_workers: (int) Number of parallel workers for file copying. Default is 0, meaning sequential execution.
    """

    relative_paths = source_directory is not None

    assert Path(target_directory).is_dir(), 'Target directory does not exist.'

    if keep_dir_structure:
        if not relative_paths:
            filepaths = pd.Series(filepaths).apply(lambda x: Path(x).relative_to(source_directory))

        parallel(copy_file_relative_to_directories, filepaths, source_directory=source_directory,
                 target_directory=target_directory, n_workers=num_workers)
    else:
        if relative_paths:
            filepaths = pd.Series(filepaths).apply(lambda x: get_absolute_path(filepath=x, root=source_directory))

        parallel(copy_file, filepaths, target_directory=target_directory)


def assemble_path(path_structure, camera_name, timestamp, set_subday_to_wildcard=False, exposure_time=None):
    """
    Assemble path by replacing timestamp, camera name and exposure time 'tags' with actual values

    :param path_structure: (str) Template for the file path. It can contain any combination of the following placeholders:
        {camera_name} for the camera name,
        {exposure_time} for the exposure time.
        {timestamp:dt_format} for the image timestamp. dt_format is a combination of the following placeholders:
                            %Y year, %m month, %d day, %H hour, %M minute, %S second (e.g. {timestamp:%Y%m%d%H%M%S})
        i.e. /a/path/to/image/{timestamp:%Y}/{camera_name}/{timestamp:%Y%m%d%H%M%S}_00{exposure_time}.jpg
    :param camera_name: (str) Name of the camera as used in image folder structure.
    :param timestamp: (datetime, tz-aware) Timestamp for which an image file is requested.
    :param set_subday_to_wildcard: (bool) If True, replace time placeholders hours, minutes and seconds with wildcards.
        Default is False. Exposure time has to be set if set_subday_to_wildcard is False.
    :param exposure_time: (int) Exposure time of images. Default is None.
    :return assembled_path: (str) Assembled file path.
    """

    if set_subday_to_wildcard:
        # replace hour with wildcard
        for replace in IGNORED_PATTERNS_SUBDAY:
            path_structure = re.sub(replace['pattern'], replace['substitution'], r'{}'.format(path_structure))

        assembled_path = path_structure.format(camera_name=camera_name, timestamp=timestamp, exposure_time=exposure_time)

    else:
        if exposure_time is None:
            assembled_path = path_structure.format(camera_name=camera_name, timestamp=timestamp)
        else:
            assembled_path = path_structure.format(camera_name=camera_name, timestamp=timestamp,
                                                  exposure_time=exposure_time)

    return assembled_path


def fstring_to_re(string):
    """
    Convert from f-string syntax to regular expression syntax

    Only a limited set of formatters supported so far, FSTRING_RE should be extended as needed.
    :param string: (str) f-string syntax string.
    :return string: (str) regular expression string.
    """
    # do not interprete '+' in the f-string as re
    string = string.replace('+', r'\+')
    # make sure not to confuse re and f-string curly brackets
    string = re.sub(r'{(.+?)}', lambda m: '__{__' + m.groups()[0] + '__}__', string)
    for replace in FSTRING_RE:
        string = re.sub(r'(__{__.*?)(' + replace['formatter'] + r')(.*?__}__)',
                        lambda m: m.groups()[0] + replace['re'] + m.groups()[2], string)
    string = re.sub(r'__{__(\w+):(.+?)__}__', lambda m: r'(?P<' + m.groups()[0] + '>' + m.groups()[1] + ')', string)
    string = re.sub(r'__{__(\w+)__}__', lambda m: r'(?P<' + m.groups()[0] + '>*)', string)

    return string
