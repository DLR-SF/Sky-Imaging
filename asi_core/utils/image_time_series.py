import asi_core.utils.datetime_handling
import asi_core.utils.filesystem
from datetime import datetime, timedelta
import os


def generate_image_series_paths(init_time, path_structure, camera_name, img_exposure_time, time_gap):
    """
    Create a list of image file paths for the given timestamp, camera, exposure time and time gaps

    :param init_time: (datetime) Initial timestamp for the creation of the image series.
    :param path_structure: (string) formated string to build up the path to each camera image folder. It shall contain
        {camera_name}, {exposure_time} and {timestamp:...} where the camera name, exposure_time and evaluated date is
        inserted.
    :param camera_name: (string) camera name
    :param img_exposure_time: (int) exposure time of the image in milliseconds.
    :param time_gap: (int) list of seconds between each image.
    :return image_paths: (string) list of image file paths
    """

    times = []
    for second in time_gap:
        times.append(init_time - timedelta(seconds=second))

    image_paths = [asi_core.utils.filesystem.assemble_path(path_structure, camera_name, current_time, exposure_time=img_exposure_time)
                   for current_time in times]

    return image_paths


def find_closest_image(missing_time, existing_paths):
    """
    Find the image file with the closest time to the missing one from the given paths list

    :param missing_time: (datetime) timestamp of the missing image file.
    :param existing_paths: (string list) list of paths to look for.
    :return: (string) path to the image file closest in time.
    """

    existing_times = [(asi_core.utils.datetime_handling.parse_datetime(p), p) for p in existing_paths]

    closest_path = min(existing_times, key=lambda et: abs(et[0] - missing_time))[1]

    return closest_path


def get_image_path(expected_path, time_tolerance):
    """
    Get the image file path with the closest time to the expected one with certain time tolerance

    :param expected_path: (string) expected path to the image file.
    :param time_tolerance: (float)  time tolerance in seconds for searching a file path.
    :return: (string) path to the image file closest in time. If nothing found return [].
    """
    closest_path = []

    base_time = asi_core.utils.datetime_handling.parse_datetime(expected_path)

    dir_path = os.path.dirname(expected_path)

    if os.path.exists(dir_path):

        candidate_paths = []

        dir_files = list(filter(lambda f: f.lower().endswith('.jpg'), os.listdir(dir_path)))

        for file_name in dir_files:

            file_path = os.path.join(dir_path, file_name)

            try:

                file_datetime = asi_core.utils.datetime_handling.parse_datetime(file_name)

                is_within_time_tolerance = abs((base_time - file_datetime).total_seconds()) <= time_tolerance

                if is_within_time_tolerance:
                    candidate_paths.append(file_path)

            except ValueError:
                continue  # parse_datetime() raises a ValueError when gets a none asi-core file. In that case do nothing

            except Exception as e:
                raise e

        if candidate_paths:
            closest_path = find_closest_image(base_time, candidate_paths)

    return closest_path


def get_image_series_path(init_time, path_structure, camera_name, exposure_time, time_gap=[0], time_tolerance=0):
    """
    Get a list of paths of a series of images given a list of seconds between each image.
    
    :param init_time: (datetime) initial timestamp for the creation of the image series.
    :param path_structure: (string) formated string to build up the path to each camera image folder. It shall contain
        {camera_name}, {exposure_time} and {timestamp:...} where the camera name, expos is inserted, evaluated date is
        inserted and where the exposure time is inserted.
    :param camera_name: (string) camera name
    :param exposure_time: (int) exposure time of the image in milliseconds.
    :param time_gap: (int) list of seconds between each image. Default is [0]
    :param time_tolerance: (float) time tolerance in seconds for searching a file path. Default is 0
    """
    expected_paths = generate_image_series_paths(init_time, path_structure, camera_name, exposure_time, time_gap)

    actual_paths = []

    for path in expected_paths:

        if os.path.exists(path):
            actual_paths.append(path)

        else:
            closest_path = get_image_path(path, time_tolerance)

            if closest_path:
                actual_paths.append(closest_path)

    return actual_paths


if __name__ == "__main__":
    print('Please run via the main program (__main__.py)!')