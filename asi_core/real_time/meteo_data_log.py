from datetime import datetime
import pandas as pd
import csv
import pvlib

import asi_core.utils.datetime_handling


class MeteoDataLog():
    """
    Abstract class of meteo data logger.
    Method get_new_data shall be implement for each type of logger
    :param log_filepath: (str) file path of the meteo data log
    :param timezone: (str) desired time zone (e.g. "GMT+1" = UTC+1)
    :param log_size: (Timedelta) size of the log express as Timedelta, default is 3 days 
    :param latitude: (float) latitude of the camera position. Default is 37.1º
    :param longitude: (float) longitude of the camera position. Default is -2.36º
    :param altitude: (float) altitude of the camera position. Default is 490m
    :param min_sun_elevation: (float) minimum sun elevation to log meteo data. Default is 5º
    :param when_to_resize_log: (time) time of day when to resize log. Default is 00:00:00 
    :param write_mode: (str) meteo data log write mode, 'w' for write or 'a' for append. Default is 'w'
    """
    
    def __init__(self, log_filepath, timezone, log_size = pd.Timedelta(days=3),
                  latitude=37.1, longitude=-2.36, altitude=490,
                  min_sun_elevation=5,
                  when_to_resize_log=datetime.strptime("00:00:00", "%H:%M:%S").time(),
                  write_mode = 'w'):
        """
        Contructor
        :param log_filepath: (str) file path of the meteo data log
        :param timezone: (str) desired time zone (e.g. "GMT+1" = UTC+1)
        :param log_size: (Timedelta) size of the log express as Timedelta, default is 3 days 
        :param latitude: (float) latitude of the camera position. Default is 37.1º
        :param longitude: (float) longitude of the camera position. Default is -2.36º
        :param altitude: (float) altitude of the camera position. Default is 490m
        :param min_sun_elevation: (float) minimum sun elevation to log meteo data. Default is 5º
        :param when_to_resize_log: (time) time of day when to resize log. Default is 00:00:00 
        :param write_mode: (str) meteo data log write mode, 'w' for write or 'a' for append. Default is 'w'
        """

        self.log_filepath = log_filepath
        self.timezone = timezone
        self.log_size = log_size
        self.latitude = latitude 
        self.longitude= longitude
        self.altitude = altitude
        self.min_sun_elevation = min_sun_elevation
        self.when_to_resize_log = when_to_resize_log
        self.write_mode = write_mode
        self.is_log_resized = False


    def add_new_data(self, new_meteodata_df):
        """
        Process new data from a campbell scientific logger
        :param new_meteodata_df: (DataFrame) New meteo data to add to the log
        """

        solar_pos = pvlib.solarposition.get_solarposition(new_meteodata_df.index[0],
                                                        self.latitude, self.longitude, self.altitude,
                                                        method='nrel_numpy')                 
        
        if all(solar_pos.elevation > self.min_sun_elevation):
        
            new_meteodata_df.to_csv(self.log_filepath, mode=self.write_mode, header=(self.write_mode == 'w'))
        
            self.write_mode = 'a'
        
        self.check_log_resize()

    def check_log_resize(self):
        """
        Check when it is time to resize the log
        """
        timestamp = datetime.now(asi_core.utils.datetime_handling.get_ETC_GMT_timezone(self.timezone))
        start = timestamp.time()
        end = (timestamp + pd.Timedelta(minutes=1)).time()

        is_time_to_resize_log = (start < self.when_to_resize_log < end) 
        
        if is_time_to_resize_log & (not self.is_log_resized):
            self.resize_log()

        self.is_log_resized = is_time_to_resize_log
        

    def resize_log(self):
        """
        Resize the log file to the given duration
        """
        df = pd.read_csv(self.log_filepath, parse_dates=["Timestamp"])
        df.set_index("Timestamp", inplace=True)
        
        log_end_time = datetime.now(asi_core.utils.datetime_handling.get_ETC_GMT_timezone(self.timezone)) - self.log_size
        df = df[df.index >= log_end_time]
        
        df.to_csv(self.log_filepath, mode='w', header=True)
    
