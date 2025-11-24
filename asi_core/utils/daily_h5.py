import glob
import os
import re
from datetime import datetime, date
from pathlib import Path

import pandas as pd
from h5py import File, Group


class DailyH5:
    """Base class for DailyH5 file manipulation"""

    def __init__(self, products_path, meta_infos={}):
        """
        Initializes a writer of daily h5 files

        :param products_path: Basepath to which daily h5 files will be stored
        :param meta_infos: Dict of meta infos valid for a whole daily h5 file. Stored once in the h5 file.
        """

        self.products_path = Path(products_path)
        self.meta_infos = meta_infos

        self.daily_h5 = {'date': date(1980, 1, 1), 'path': None}

    def get_file(self, timestamp):
        """
        Get the path to the current daily h5 file. Initialize if not done yet.

        :param timestamp: Timestamp of the data to be stored
        :return: Path to the current daily h5 file
        """

        if self.daily_h5['date'] != timestamp.date():
            self.daily_h5 = {'date': date(1980, 1, 1), 'path': None}
            self.init_h5file(timestamp)

        return self.daily_h5['path']

    def init_h5file(self, timestamp):
        """
        Initialize daily h5 file for reading or writing.

        :param timestamp: Timestamp of the current data to be stored
        """
        raise Exception('Not implemented')

    def process_entry(self, timestamp, mode, data=None, timestamp_forecasted=None):
        """
        Stores a dataset of one timestamp to the daily h5 file

        :param timestamp: Timestamp based on which dataset was created
        :param mode: character, r/w/a, i.e. read, write or append
        :param data: Dataset to be saved either dataset which can be stored by h5py or dict of such datasets
        :param timestamp_forecasted: (Optional) timestamp forecasted by the dataset
        """

        target_file = self.get_file(timestamp)
        label_entry = f'{timestamp:%H%M%S}'
        if timestamp_forecasted is not None:
            if timestamp_forecasted.date() != timestamp.date():
                raise Exception('Only intraday forecasts expected!')

            label_entry += f'_{timestamp_forecasted:%H%M%S}'

        with File(target_file, mode) as f:
            data_out = self.process_sub_entry(f, label_entry, data)
        return data_out

    def process_sub_entry(self, label, data=None):
        """
        Defines the read/ write operation to be applied recursively

        :param label: Label of the current data to be stored/ read
        :param data: Data to be processed
        """
        raise Exception('Not implemented')


class DailyH5Writer(DailyH5):

    def store_entry(self, timestamp, data, timestamp_forecasted=None):
        """
        Stores a dataset of one timestamp to the daily h5 file

        :param timestamp: Timestamp based on which dataset was created
        :param timestamp_forecasted: (Optional) timestamp forecasted by the dataset
        :param data: Dataset to be saved either dataset which can be stored by h5py or dict of such datasets
        """
        self.process_entry(timestamp, 'a', data=data, timestamp_forecasted=timestamp_forecasted)

    def init_h5file(self, timestamp, do_not_overwrite=True):
        """
        Initialize daily h5 file, create folders if required, store meta infos to a new h5 file.

        :param timestamp: Timestamp of the current data to be stored
        :param do_not_overwrite: If called and a daily file already exists, create additional file instead of
                                 overwriting the previous one.
        """

        if not os.path.isdir(self.products_path):
            os.makedirs(self.products_path)

        file_counter = 0
        h5_path = str(self.products_path / f'{timestamp:%Y%m%d}')
        if do_not_overwrite:
            file_counter = len(glob.glob(h5_path + '*.h5'))
        if file_counter:
            h5_path += f'_{file_counter+1}'
        h5_path += '.h5'

        with File(h5_path, 'w') as f:
            data = self.process_sub_entry(f, 'meta', self.meta_infos)

        self.daily_h5 = {'date': timestamp.date(), 'path': h5_path}

    def process_sub_entry(self, handle, label, data):
        """
        Recursively store all datasets in data

        :param handle: Handle to an h5file or a group in an h5 file
        :param label: Label under which data will be stored
        :param data: dataset or dict of datasets
        """

        if type(data) is dict:
            handle.create_group(label)
            for k, v in data.items():
                self.process_sub_entry(handle[label], k, v)
        else:
            handle[label] = data
        return None


class DailyH5Reader(DailyH5):
    @staticmethod
    def list_entries(h5_path):
        """
        Generate a dataframe of the keys and corresponding timestamps and forecasted timestamps in the h5 file.

        :param h5_path: Path of the h5 file, the ke
        :return: Dataframe with columns key, timestamp, forecasted_timestamp
        """

        filename = Path(h5_path).stem

        entries = pd.DataFrame({'key': [], 'timestamp': [], 'timestamp_forecasted': []})

        with File(h5_path, 'r') as f:
            for k in f.keys():
                timestamp = None
                forecasted_timestamp = None
                fmt_1 = re.search(r'(\d{6})_(\d{6})', k)
                fmt_2 = re.search(r'(\d{6})', k)
                if type(fmt_1) is re.Match:
                    timestamp = datetime.strptime(filename + fmt_1.groups()[0], '%Y%m%d%H%M%S')
                    forecasted_timestamp = datetime.strptime(filename + fmt_1.groups()[1], '%Y%m%d%H%M%S')
                elif type(fmt_2) is re.Match:
                    timestamp = datetime.strptime(filename + fmt_2.groups()[0], '%Y%m%d%H%M%S')
                new_entry = pd.DataFrame({'key': [k], 'timestamp': [timestamp],
                                          'timestamp_forecasted': [forecasted_timestamp]})
                if not entries.empty:
                    entries = pd.concat([entries, new_entry], ignore_index=True)
                else:
                    entries = new_entry
        return entries

    @staticmethod
    def init_from_path(timestamp, h5_path):
        """
        Create a DailyH5Reader instance and initializes it from a specific h5 file

        :param timestamp: Timestamp based on which dataset was created
        :param h5_path: Path of the h5 file to be read (naming can deviate from convention of DailyH5Writer)
        :return: DailyH5Reader instance
        """

        reader = DailyH5Reader('')
        with File(h5_path, 'r') as f:
            reader.meta_infos = reader.process_sub_entry(f, 'meta')

        reader.daily_h5 = {'date': timestamp.date(), 'path': h5_path}
        return reader

    def get_entry(self, timestamp, timestamp_forecasted=None):
        """
        Stores a dataset of one timestamp to the daily h5 file

        :param timestamp: Timestamp based on which dataset was created
        :param timestamp_forecasted: (Optional) timestamp forecasted by the dataset

        :return: Dataset to be saved either dataset which can be stored by h5py or dict of such datasets
        """
        return self.process_entry(timestamp, 'r', timestamp_forecasted=timestamp_forecasted)

    def process_sub_entry(self, handle, label, data=None):
        """
        Recursively get all datasets in handle

        :param handle: Handle to an h5file or a group in an h5 file
        :param label: Label under which data will be stored
        :param data: dataset or dict of datasets
        """
        sub_handle = handle[label]
        if type(sub_handle) is Group:
            data = {}
            for k in sub_handle.keys():
                data[k] = self.process_sub_entry(sub_handle, k)
        else:
            data = sub_handle[()]
        return data

    def init_h5file(self, timestamp, file_counter=0):
        """
        Initialize daily h5 file, load meta data.

        :param timestamp: Timestamp of the current data to be stored
        :param file_counter: Appends counter suffix to file name. May be useful if multiple files created for a day.
        """

        h5_path = str(self.products_path / f'{timestamp:%Y%m%d}')
        if file_counter:
            h5_path += f'_{file_counter+1}'
        h5_path += '.h5'

        with File(h5_path, 'r') as f:
            self.meta_infos = self.process_sub_entry(f, 'meta')

        self.daily_h5 = {'date': timestamp.date(), 'path': h5_path}
