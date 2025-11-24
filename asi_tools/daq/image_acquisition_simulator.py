# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
This script serves as a simulation for a camera real-time data acquisition, based on historical images.
This is intended to check the real-time operation of various applications.

Some user inputs have to be defined (see #user input)
"""

# Import necessary libraries/modules
import os
import shutil
import sched
import time
import datetime


def copy_periodically(scheduler, interval_seconds, start_time, end_time, origin_folder, destination_folder,
                      call_count, add_path_month_day):
    """
    Main function which check for the images, creates a copy and schedules the next periodical call.

    :param scheduler: sched.scheduler object
    :param interval_seconds: used interval in seconds of scheduler -> should be according to the camera update rate
    :param start_time: datetime object which defines the first time stamp
    :param end_time: datetime object which defines the last time stamp
    :param origin_folder: main folder to historical images'
    :param destination_folder: main folder for the simulated image acquisition
    :param call_count: index which takes track of already evaluated images
    :param add_path_month_day: zero padded string for month and day folder
    """

    current_time = start_time + datetime.timedelta(seconds=interval_seconds*call_count[0])
    # Check if the end_time is reached
    if current_time > end_time:
        print("End time reached. Stopping the timer.")
        return
    current_hour_of_day = current_time.strftime("%H")

    files = os.listdir(origin_folder + '/' + add_path_month_day + '/' + current_hour_of_day)
    images = [f for f in files if f.endswith(('.jpg', '.png', '.jpeg', '.gif', '.bmp'))]
    if images:
        datetime_list_images = [datetime.datetime.strptime(image[:14], "%Y%m%d%H%M%S") for image in images]

        (closest_datetime, position_closest_datetime) = find_closest_datetime(datetime_list_images, current_time)
        deviation_time_image = abs(closest_datetime - current_time)

        if deviation_time_image < datetime.timedelta(seconds=5):
            full_path_image = (origin_folder + add_path_month_day + '/' + current_hour_of_day +
                               '/' + images[position_closest_datetime])
            full_path_destination_folder = (destination_folder + add_path_month_day + '/' + current_hour_of_day)
            full_path_destination_image = (full_path_destination_folder + '/' + images[position_closest_datetime])

            print('Copy image from ' + full_path_image + '\n' + ' to ' + full_path_destination_image)
            # Create the destination folder if it doesn't exist
            os.makedirs(full_path_destination_folder, exist_ok=True)
            try:
                # Copy the image
                shutil.copy(full_path_image, full_path_destination_image)
            except Exception as e:
                print(f"An error occurred: {str(e)}")
        else:
            print('no valid image for ' + current_time.strftime("%Y.%m.%d %H:%M:%S") )
    else:
        print('no images in ' + origin_folder + '/' + add_path_month_day + '/' + current_hour_of_day)

    # Update the call count
    call_count[0] += 1

    current_real_time, delay = calculate_deviation(interval_seconds)
    # Schedule the next copy_periodically call
    scheduler.enter(delay, 1, copy_periodically,
                    (scheduler, interval_seconds, start_time, end_time, origin_folder, destination_folder, call_count,
                     add_path_month_day))


def calculate_deviation(interval_seconds):
    """
    This function calculates the temporal deviation from the current time to the next expected image acquisition.

    :param interval_seconds: used interval in seconds of scheduler -> should be according to the camera update rate
    :returns: datetime object of the current real computer time -> only used to keep track of update rate
    :returns: temporal deviation in seconds for the next expected image acquisition event.
    """
    current_time = datetime.datetime.now()
    time_in_seconds = current_time.second + current_time.microsecond / 1e6  # Convert microseconds to seconds
    seconds_since_last_interval = time_in_seconds % interval_seconds
    deviation = interval_seconds - seconds_since_last_interval
    return current_time, deviation


def find_closest_datetime(datetime_list, target_datetime):
    """
    This function finds from the historical available images to the closest match to the expected time stamp.
    The real camera acquisition does not work perfectly. Some images are created with a delay of several seconds,
    which is reflected in the name of the images.

    :param datetime_list: list of datetime objects with the available images within the current hour folder.
    :param target_datetime: datetime object with the expected time stamp of the next image
    :returns: datetime object of the image closest to the target time
    :returns: position of the image within the list corresponding to the determined closest_datetime
    """
    closest_datetime = min(datetime_list, key=lambda dt: abs(dt - target_datetime))
    position = datetime_list.index(closest_datetime)
    return closest_datetime, position


if __name__ == "__main__":
    # --------------------------------
    # user input
    origin_folder = "//129.247.24.3/Meteo/MeteoCamera/2020/Cloud_Cam_Metas"
    destination_folder = "C:/Nouri/Meteo/PyranoCam/TestImageAcquisitionSimulator"
    # start_time and end_time have to be within the same day
    start_time = datetime.datetime.strptime("20200607165800", "%Y%m%d%H%M%S")
    end_time = datetime.datetime.strptime("20200607170100", "%Y%m%d%H%M%S")
    # define the update rate of the image acquisition
    interval_seconds = 30  # Change this to your desired interval in seconds
    # --------------------------------

    # raise error if time window is invalid (multiple days are not permitted)
    if start_time.date() != end_time.date():
        raise ValueError("The start time and end time are not from the same day. Check your user input!")
    # start scheduler for data acquisition
    call_count = [0]  # Initialize the call count -> takes track of already evaluated images
    # create
    add_path_month_day = start_time.strftime("/%m/%d")

    s = sched.scheduler(time.time, time.sleep)
    # start at next full minute
    current_real_time, delay = calculate_deviation(60)
    print(f'Scheduler for image acquisition process starts in {delay:.2f} s')
    s.enter(delay, 1, copy_periodically, (s, interval_seconds, start_time, end_time, origin_folder, destination_folder,
                                          call_count, add_path_month_day))
    s.run()
