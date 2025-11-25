import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd

import asi_core.utils.datetime_handling
from asi_core.real_time.meteo_data_log import MeteoDataLog

class MeteoDataWebLog(MeteoDataLog):
    """
    Meteo Data Logger of html string of a campbell scientific logger

    :param url_cs_logger_table: (str) url of table as string
    :param log_filepath: (str) file path of the meteo data log
    :param timezone: (str) desired time zone (e.g. "GMT+1" = UTC+1)
    :param name_desired_columns_cs_table: (dict) List with 2 columns and n rows. First column holds the original names
        of channels which shall be renamed. The second column holds the new names for the channels.
    :param log_size: (Timedelta) size of the log express as Timedelta, default is 3 days 
    :param latitude: (float) latitude of the camera position. Default is 37.1º
    :param longitude: (float) longitude of the camera position. Default is -2.36º
    :param altitude: (float) altitude of the camera position. Default is 490m
    :param min_sun_elevation: (float) minimum sun elevation to log meteo data. Default is 5º
    :param when_to_resize_log: (time) time of day when to resize log. Default is 00:00:00 
    :param write_mode: (str) meteo data log write mode, 'w' for write or 'a' for append. Default is 'w'
    """

    def __init__(self, url_cs_logger_table, log_filepath, timezone, name_desired_columns_cs_table=None, 
                 log_size = pd.Timedelta(days=3),
                 latitude=37.1, longitude=-2.36, altitude=490,
                 min_sun_elevation=5,
                 when_to_resize_log=datetime.strptime("00:00:00", "%H:%M:%S").time(),
                 write_mode = 'w'):
        """
        Contructor

        :param url_cs_logger_table: (str) url of table as string
        :param log_filepath: (str) file path of the meteo data log
        :param timezone: (str) desired time zone (e.g. "GMT+1" = UTC+1)
        :param name_desired_columns_cs_table: (dict) List with 2 columns and n rows. First column holds the original
            names of channels which shall be renamed. The second column holds the new names for the channels.
        :param log_size: (Timedelta) size of the log express as Timedelta, default is 3 days 
        :param latitude: (float) latitude of the camera position. Default is 37.1º
        :param longitude: (float) longitude of the camera position. Default is -2.36º
        :param altitude: (float) altitude of the camera position. Default is 490m
        :param min_sun_elevation: (float) minimum sun elevation to log meteo data. Default is 5º
        :param when_to_resize_log: (time) time of day when to resize log. Default is 00:00:00 
        :param write_mode: (str) meteo data log write mode, 'w' for write or 'a' for append. Default is 'w'
        """

        super().__init__(log_filepath, timezone, log_size, latitude, longitude, altitude,
                         min_sun_elevation, when_to_resize_log, write_mode)

        self.url_cs_logger_table = url_cs_logger_table
        self.name_desired_columns_cs_table = name_desired_columns_cs_table

    def log_html_meteodata(self):
        """
        Monitor the given url and log updates in the given file
        """
        
        html_string = ""

        while True:

            response = requests.get(self.url_cs_logger_table)
            
            if response.status_code == 200:
                
                if html_string != response.text:
                    
                    html_string = response.text

                    meteodata_df = self.parse_logger_data(html_string)

                    self.add_new_data(meteodata_df)
                    

            else:
                print(f"Failed to retrieve data from CS logger. Status code: {response.status_code}")


    def parse_logger_data(self, html_string):
        """
        Parse a html string of a campbell scientific logger

        :param html_string: (str) html string of a campbell scientific logger
        :return meteodata_df: data frame with n columns for each channel + timestamp index as datetime
        """

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_string, 'html.parser')

        # Find the data within the table
        table = soup.find('table')
        rows = table.find_all('tr')

        # Initialize lists to store data
        columns = []
        values = []
        for row in rows:
            # Extract column names (headers) and values
            th = row.find('th')
            td = row.find('td')

            if th and td:
                column = th.get_text()
                value = float(td.get_text())  # Convert value to float if needed

                columns.append(column)
                values.append(value)

        # get time stamp
        b_element = soup.find('b', string='Record Date: ')
        timestamp = b_element.find_next_sibling(string=True)
        timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')
        timestamp = pd.Timestamp(timestamp).tz_localize(tz=asi_core.utils.datetime_handling.get_ETC_GMT_timezone(self.timezone))
        
        # Create a DataFrame
        data_dict = {'Timestamp': [timestamp]}
        data_dict.update({col: [value] for col, value in zip(columns, values)})

        meteodata_df = pd.DataFrame(data_dict)
        meteodata_df.set_index("Timestamp", inplace=True)

        if self.name_desired_columns_cs_table is not None:
            logger_keys = list(self.name_desired_columns_cs_table["header_logger"])
            pyranocam_values = list(self.name_desired_columns_cs_table["header_actual"])
            columns_mapping = list(zip(logger_keys, pyranocam_values))
            for old_name, new_name in columns_mapping:
                meteodata_df.rename(columns={old_name: new_name}, inplace=True)

        return meteodata_df



if __name__ == '__main__':

    url_cs_logger_table = "http://10.21.202.135/?command=NewestRecord&table=WattsCorrSec"
    timezone = 'GMT+1'
    name_desired_columns_cs_table =  {
            "header_logger" : ["CMP21_211474_corr_Avg", "CHP1_210948_corr_Avg", "CMP21_090280_corr_Avg"],
            "header_actual" : ["dhi", "dni", "ghi"]
            }

    aLog = MeteoDataWebLog(url_cs_logger_table, "test_meteodata_obj.csv", timezone, name_desired_columns_cs_table=name_desired_columns_cs_table)

    aLog.log_html_meteodata()
        
    