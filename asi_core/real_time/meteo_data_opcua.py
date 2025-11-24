from datetime import datetime
import pandas as pd
import csv
from opcua import Client
import cryptography

import asi_core.utils.datetime_handling
from asi_core.real_time.meteo_data_log import MeteoDataLog

class MeteoDataOPCLog(MeteoDataLog):
    """
    Meteo Data Logger of OPC UA server.
    :param url_opc: (str) url of the OPC server
    :param log_meteodata_filepath: (str) file path of the meteo data log
    :param timezone: (str) desired time zone (e.g. "GMT+1" = UTC+1)
    :param log_size: (Timedelta) size of the log express as Timedelta, default is 3 days 
    :param latitude: (float) latitude of the camera position. Default is 37.1º
    :param longitude: (float) longitude of the camera position. Default is -2.36º
    :param altitude: (float) altitude of the camera position. Default is 490m
    :param min_sun_elevation: (float) minimum sun elevation to log meteo data. Default is 5º
    :param when_to_resize_log: (time) time of day when to resize log. Default is 00:00:00 
    :param write_mode: (str) meteo data log write mode, 'w' for write or 'a' for append. Default is 'w'
    """

    def __init__(self, url_opc, log_filepath, timezone, 
                 log_size = pd.Timedelta(days=3),
                 latitude=37.1, longitude=-2.36, altitude=490,
                 min_sun_elevation=5,
                 when_to_resize_log=datetime.strptime("00:00:00", "%H:%M:%S").time(),
                 write_mode = 'w'):
        """
        Constructor
        :param url_opc: (str) url of the OPC server
        :param log_meteodata_filepath: (str) file path of the meteo data log
        :param timezone: (str) desired time zone (e.g. "GMT+1" = UTC+1)
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

        self.url_opc = url_opc

    def log_opc_meteodata(self):
        """
        Monitor the given url and log updates in the given file
        """
                
        client = Client(url)

        try:

            client.connect()

            while True:            
                
                now = datetime.now()
                seconds_to_wait = 60 - now.seconds
                sleep(seconds_to_wait)

                node = client.get_node("ns=5;s=0WTV01CR001//DirSolStrl.U")
                timestamp = datetime.now(asi_core.utils.datetime_handling.get_ETC_GMT_timezone(self.timezone))
                dni = node.get_value()    

                meteodata_df = pd.DataFrame(data={"Timestamp":[timestamp], "dni":[dni]})

                self.add_new_data(meteodata_df)

        finally:
            client.disconnect()


if __name__ == '__main__':

    url = "opc.tcp://192.168.0.102:48050"
    timezone = 'GMT+1'

    log = MeteoDataOPCLog(url, "test_meteodata.csv", timezone)

    log.log_opc_meteodata()

        
    