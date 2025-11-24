# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""Tools useful to download datasets from public/remote sources"""

import requests


def download_and_store_dataset(url, targetpath_dataset):
    """Download a dataset from a url and save to file

    This tool has only been tested with text files so far.

    :param url: URL which starts the download
    :type url: str
    :param targetpath_dataset: Path including filename to which dataset should be stored
    :type targetpath_dataset: str, pathlib.Path
    """

    response = requests.get(url, stream=True)

    with open(targetpath_dataset, "wb") as handle:
        for data in response.iter_content(chunk_size=1024):
            handle.write(data)
