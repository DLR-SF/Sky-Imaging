import glob
from calendar import monthrange
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import numpy as np

import pvlib
from datetime import datetime, timedelta, date

import dateutil.tz
import sys

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


class AvailabilityChecker():
    def __init__(self, img_base_path, cam_id):
        """
        Initialize the tool to evaluate and plot image availability.
        
        The tool assumes the "hourly" folder structure. get_img_path can be adapted in a sub-class to modify this.
        
        Args:
            img_base_path -- path in which images of the evaluated camera and year are stored.
            cam_id -- string, identifier of the camera as indicated in image folder path
        """
        self.img_base_path = img_base_path
        self.cam_id = cam_id

    def get_img_path(self, day_eval):
        """
        Get the path to requested timestamp and station.
        This function is specific to the hourly folder structure. It may be overridden in the future.
        """
        return fr'{self.img_base_path}\{day_eval:%Y}\{self.cam_id}\{day_eval:%m}\{day_eval:%d}\*\{day_eval:%Y%m%d}*_*.jpg'

    def get_availability(self, year, month, tz_offset, station_coords, sampling_time):
        """
        Compare expected timestamps with the found timestamps and compile image availability in dictionary.

        Args:
            year -- string (%Y), year for which image availability is evaluated
            month -- string (%m), month for which image availability is evaluated
            tz_offset -- integer, offset in hours from UTC
            station_coords -- dictionary with keys lat (latitude northing in float/ decimal degrees), 
                lon (longitude easting in float/ decimal degrees)
            sampling_time -- float [seconds], sampling time

        Returns:
            availabilityInfo -- dictionary, informing on image availability. Keys:
                num_imgs_per_day_expected -- 1D array of integers, number of expected images based on sun position
                num_imgs_per_day_found -- 1D array of integers, number of found images based on sun position
                histogram_days_hours -- numpy 2D array, histogram with days on y and sampling intervals on x
                year -- string (%Y), year for which image availability is evaluated
                month -- string (%m), month for which image availability is evaluated
        """

        # Dates
        num_days_in_month = monthrange(year, month)[1]  # num_days = 28
        first_day = datetime(year, month, 1, 0, 0, 0, tzinfo=dateutil.tz.tzoffset(None, tz_offset * 3600))
        last_day = datetime(year, month, num_days_in_month, 0, 0, 0,
                                  tzinfo=dateutil.tz.tzoffset(None, tz_offset * 3600))

        # Get time of sunrise and sunset
        tm = pd.date_range(start=first_day.strftime('%Y-%m-%d'), end=last_day.strftime('%Y-%m-%d'),
                           freq='1D', tz=dateutil.tz.tzoffset(None, tz_offset * 60 * 60))
        sun_position = pvlib.solarposition.sun_rise_set_transit_spa(times=tm, latitude=station_coords['lat'],
                                                                    longitude=station_coords['lon'], how='numba',
                                                                    delta_t=69.3, numthreads=4)
        # creating a list which every day per month
        days_eval = [first_day + timedelta(days=dayi) for dayi in range(num_days_in_month)]

        frames_per_hour = int(3600/sampling_time)

        # preparing a numpy array, first filled with 'NaN'
        availability = np.ones((num_days_in_month, 24 * frames_per_hour)) * np.nan
        found_timestamps = []
        for [d, day_eval] in enumerate(days_eval):
            expected_timestamps = [day_eval + timedelta(seconds=int(secs)) for secs in np.arange(0,3600*24+sampling_time, sampling_time)]
            found_imgs = glob.glob(self.get_img_path(day_eval))
            for img in found_imgs:
                patterns = img.split('\\')
                found_timestamps.append(pd.to_datetime(patterns[-1][:14]).replace(tzinfo=dateutil.tz.tzoffset(None, tz_offset * 60 * 60)))

            histogram_counts, _ = np.histogram(found_timestamps, bins=expected_timestamps)
            availability[d, :] = histogram_counts
            availability[d, [t < sun_position['sunrise'][d] or t > sun_position['sunset'][d]
                             for t in expected_timestamps[:-1]]] = np.nan

        availability = np.flipud(availability)

        num_target_list = []
        num_is_list = []
        for each_day in range(len(days_eval) - 1, -1, -1):
            num_total = 24 * frames_per_hour
            num_nan = np.count_nonzero(np.isnan(availability[each_day]))
            num_target = num_total - num_nan
            num_target_list.append(num_target)
            num_is = (availability[each_day] > 0).sum()
            num_is_list.append(num_is)

        availability_info = {'num_imgs_per_day_expected': num_target_list, 'num_imgs_per_day_found': num_is_list,
                             'histogram_days_hours': availability, 'year': year, 'month': month}
        return availability_info

    def plot_availability(self, availability_info):
        """
        Plot image availability for the evaluated month.

        Args:
            availabilityInfo -- dictionary, informing on image availability. Keys:
                num_imgs_per_day_expected -- 1D array of integers, number of expected images based on sun position
                num_imgs_per_day_found -- 1D array of integers, number of found images based on sun position
                histogram_days_hours -- numpy 2D array, histogram with days on y and sampling intervals on x
                year -- string (%Y), year for which image availability is evaluated
                month -- string (%m), month for which image availability is evaluated
        """

        histogram_days_hours = availability_info['histogram_days_hours']
        num_target_list = availability_info['num_imgs_per_day_expected']
        num_is_list = availability_info['num_imgs_per_day_found']
        year = availability_info['year']
        month = availability_info['month']

        xmin, xmax = mdates.datestr2num(['2022-04-01', '2022-04-02'])
        fig, ax = plt.subplots(figsize=(7,4))
        month_name = datetime.strptime(str(month), "%m").strftime("%B")
        number_days = np.shape(histogram_days_hours)[0]
        plt.title(f'Availability of ASI images in {month_name} {year}', fontsize=12, fontweight ='bold', loc='left', pad=10)
        im = ax.imshow(histogram_days_hours, aspect='auto', extent=(xmin, xmax, 0.5, number_days + 0.5), interpolation='None', vmax=2)
        ax.xaxis_date()
        date_format = mdates.DateFormatter('%H:%M')
        ax.set_xlabel('Hour of the day', fontweight ='bold')
        ax.xaxis.labelpad = 20
        ax.set_ylabel('Day of the Month', fontweight = 'bold')
        ax.yaxis.labelpad = 10
        ax.xaxis.set_major_formatter(date_format)
        ax.tick_params(axis='x', labelsize=8, labelrotation=90, length=5)
        ax.tick_params(axis='y', labelsize=8)
        fig.tight_layout()
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
        major_ticks = np.arange(1, number_days + 1, 1)
        minor_ticks = np.arange(0.5, number_days + 0.5, 1)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        ax.grid(which='major', alpha=0.5, linestyle='dotted')
        ax.grid(which='minor', alpha=1)


        secax = ax.secondary_yaxis('right')
        secax.set_yticks(major_ticks)
        secax.tick_params(axis='y', labelsize=8)

        labels = []
        for item in secax.get_yticklabels():
            labels.append(item.get_text())
        for labelnum in range(0, len(labels)):
            labels[labelnum] = f'{num_is_list[labelnum]}/{num_target_list[labelnum]}'

        secax.set_yticklabels(labels)
        secax.set_ylabel('# available images / # expected images', fontweight = 'bold')
        secax.yaxis.labelpad = 20
        cbar = plt.colorbar(im, orientation='vertical', ticks=[0, 1, 2], boundaries=[-0.5, 0.5, 1.5, 2.5], pad=0.2)
        cbar.ax.set_yticklabels(['0', '1', '> 1'])
        cbar.set_label('# images in sampling interval', fontweight = 'bold')

        plt.savefig(fr'{self.img_base_path}\{year:04}\{self.cam_id}\{month:02}\ASI_availability.png')


if __name__ == '__main__':
    # this should come directly from the stations config file in the future!
    tz_offset = int(sys.argv[1])
    sampling_time_imgs = int(sys.argv[2])
    img_base_path = sys.argv[3]
    cam_id = sys.argv[4]
    station_coords = {'lat': float(sys.argv[5]), 'lon': float(sys.argv[6])}

    if len(sys.argv) == 8:
        year_eval = int(sys.argv[6])
        month_eval = int(sys.argv[7])
    else:
        yesterday = date.today() - timedelta(days=1)
        year_eval = yesterday.year
        month_eval = yesterday.month

    ac = AvailabilityChecker(img_base_path, cam_id)

    availability_info = ac.get_availability(year_eval, month_eval, tz_offset, station_coords, sampling_time_imgs)
    ac.plot_availability(availability_info)
