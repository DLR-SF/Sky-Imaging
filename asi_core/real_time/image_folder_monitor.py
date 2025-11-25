"""
Main script, multiple functions and one class in order to monitor folders for new images. New images trigger event:
e.g. real time camera application. If desired results are presented over a web application.
Classes: ImageHandler: child class from FileSystemEventHandler (watchdog.events) which monitors the event of new images.
"""
import time
import os
import subprocess

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime

import asi_core.utils.datetime_handling
import asi_core.utils.filesystem
from asi_core.utils import image_time_series


class ImageHandler():
    """
    A class for handling image file creation events and processing image files.
    """

    def __init__(self, on_event_do_function, exposure_time):
        super().__init__()
        self.on_event_do_function = on_event_do_function
        self.exposure_time = exposure_time        

    def process_new_image(self, new_filepath):
        """
        React to file creation events if the created file is an image ("jpg", "jpeg", "png")
        :param new_filepath: (str) new file path
        """
        
        file_extension = new_filepath.split(".")[-1].lower()
        img_exposure_time = self.extract_exposure_time(new_filepath)

        if file_extension in ["jpg", "jpeg", "png"] and img_exposure_time in self.exposure_time:
            self.on_event_do_function(new_filepath)


    def extract_exposure_time(self, image_filepath):
        """
        Extract exposure time from a file name.
        :param image_filepath: (string) path to the image file
        :return exposure_time: (int) exposure time.
        """
        exposure_time = -1

        filename = os.path.basename(image_filepath)
        exposure_time_str = filename[filename.find("_")+1 : filename.find(".")]

        try:
            exposure_time = int(exposure_time_str)

        except ValueError as e:
            print("No exposure time found for this file")

        return exposure_time
        
        
class ImageWatchdog(FileSystemEventHandler):
    """
    A class for handling file creation events and processing image files.
    This class extends the functionality of the `FileSystemEventHandler` class by
    reacting to file creation events.
    """

    def __init__(self, an_Image_Handler):
        super().__init__()
        self.Image_Handler = an_Image_Handler


    def on_created(self, event):
        """
        React to file creation events. 
        :param event: (object) event triggered by watchdog
        """

        if not event.is_directory:
            
            self.Image_Handler.process_new_image(event.src_path)


def start_image_watchdog(on_event_do_function, path_to_watch_structure, camera_name, exposure_time, timezone):
    """
    React to image file ("jpg", "jpeg", "png") creation events running the given function.

    :param on_event_do_function: (function) function to run when a new image is created. This function shall get a path
        to the created image file as the only input argument.
    :param path_to_watch_structure: (string) path to watch folder, containing {camera_name} where the camera name should
        be inserted and {timestamp:...} (e.g. {timestamp:%Y%m%d%H%M%S%f}) where the evaluated timestamp should be inserted
    :param camera_name: (string) name of the camera.
    :param exposure_time: (int) list of exposure time of the image in milliseconds.
    :param timezone: (timezone) time zone of the datimes in the file 
    """

    init_time = datetime.now(asi_core.utils.datetime_handling.get_ETC_GMT_timezone(timezone))
    path_to_watch = asi_core.utils.filesystem.assemble_path(path_to_watch_structure, camera_name, init_time, exposure_time=exposure_time)

    handler = ImageHandler(on_event_do_function, exposure_time)
    watchdog = ImageWatchdog(handler)

    observer = Observer()
    observer.schedule(watchdog, path_to_watch, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()

def get_file_list(path):
    """
    Get the list of all files in the given path
    :param path: (str) path to get the list of files.
    :return file_list: (list str) list of files in the given path.
    """
    file_list = []

    for root, dirs, files in os.walk(path):
        for file in files:
            file_list.append(os.path.join(root, file))

    return file_list

def start_image_polling(on_event_do_function, path_to_watch_structure, camera_name, exposure_time, timezone, polling_interval=2):
    """
    React to image file ("jpg", "jpeg", "png") creation running the given function.

    :param on_event_do_function: (function) function to run when a new image is created. This function shall get a path
        to the created image file as the only input argument.
    :param path_to_watch_structure: (string) path to watch folder, containing {camera_name} where the camera name should
        be inserted and {timestamp:...} (e.g. {timestamp:%Y%m%d%H%M%S%f}) where the evaluated timestamp should be inserted
    :camera_name: (string) name of the camera.
    :param exposure_time: (int) list of exposure time of the image in milliseconds.
    :param timezone: (timezone) time zone of the datimes in the file 
    :polling_interval: (int) time to wait between polling checks in seconds. Default is 2.
    """

    handler = ImageHandler(on_event_do_function, exposure_time)

    init_time = datetime.now(asi_core.utils.datetime_handling.get_ETC_GMT_timezone(timezone))
    path_to_watch = asi_core.utils.filesystem.assemble_path(path_to_watch_structure, camera_name, init_time, exposure_time=exposure_time)

    previous_files = get_file_list(path_to_watch)

    while True:
        
        time.sleep(polling_interval)

        init_time = datetime.now(asi_core.utils.datetime_handling.get_ETC_GMT_timezone(timezone))
        path_to_watch = asi_core.utils.filesystem.assemble_path(path_to_watch_structure, camera_name, init_time, exposure_time=exposure_time)
        current_files = get_file_list(path_to_watch)

        new_files = [value for value in current_files if value not in previous_files]

        if new_files:

            for new_file in new_files:
                
                handler.process_new_image(new_file)
                
                previous_files = current_files


def run_image_folder_monitor(on_event_do_function, path_to_watch_structure, camera_name, exposure_time, timezone='UTC', use_watchdog=True, polling_interval=2):
    """
    React to image file ("jpg", "jpeg", "png") creation events running the given function.

    :param on_event_do_function: (function) function to run when a new image is created. This function shall get a path
        to the created image file as the only input argument.
    :param path_to_watch_structure: (string) path to watch folder, containing {camera_name} where the camera name should
        be inserted and {timestamp:...} (e.g. {timestamp:%Y%m%d%H%M%S%f}) where the evaluated timestamp should be inserted
    :camera_name: (string) name of the camera.
    :param exposure_time: (int) list of exposure time of the image in milliseconds.
    :param timezone: (timezone) time zone of the datetimes in the file. Default is UTC.
    :param use_watchdog: (boolean) True for using watchdog, False for using polling. Defaults is True.
    :param polling_interval: (int) time to wait between polling checks in seconds. Default is 2.
    """

    if use_watchdog:
        start_image_watchdog(on_event_do_function, path_to_watch_structure, camera_name, exposure_time, timezone)
    
    else:
        start_image_polling(on_event_do_function, path_to_watch_structure, camera_name, exposure_time, timezone, polling_interval)

if __name__ == '__main__':

    print('Please run via the main program (__main__.py)!')
