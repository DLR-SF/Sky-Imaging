# Quick start guide on HTTP image receiver data acquisition program

## Requirements

This DAQ needs to be installed on a computer (linux or windows) located in the same network as the 
camera you want to control. (The camera must be accessible e.g. in your browser.)

Make sure that the asi-core package is properly installed. 

To learn how to setup asi-core see [Getting started](getting_started.md)


## Set your local configuration

- Create config file for your data acquisition task based on this template

  `<path to repository>\data\camera_data\ASI_Template.yaml`

- Set the name / id of the camera which should be used uniformly in folder names and all configurations e.g.

  `camera_name: 'Cloud_Cam_Metas'`

- Set the model of the camera e.g.

  `camera_model: 'Q25'`

- Set the coordinates of the camera e.g.

  `latitude: 37.091573`

  `longitude: -2.363595`

  `altitude: 500.`

- Set the URL of the camera in your local network e.g.

  `daq/url_cam: 'http://10.21.202.145'`

 
- Set the path to which you want to store your results:
 
  `daq/storage_path: 'C:/data/test_daq/server_folder'`


- Set the working directory of the DAQ program. In this folder log files will be stored
 
  `daq/daq_working_dir: 'C:/data/test_daq/asi-core/log_folder'`


## Run the DAQ

- Activate the conda environment 

  `conda activate asi_core`


- Run the daq providing your config file as argument, e.g.:

  `python C:\git\sfpt_meteo_nowcasting\asi-core\asi_core\daq\http_image_receiver.py -c "C:\git\sfpt_meteo_nowcasting\asi-core\data\camera_data\Cloud_Cam_Metas_20190711.yaml"`


- You can adapt the following script to do the job

  `C:\git\sfpt_meteo_nowcasting\asi-core\asi_core\daq\run_all_GUIs.bat`

- We recommend to restart the DAQ once per day. E.g. around midnight.


## Further settings

The DAQ uses a number of default settings which can be changed if needed when initializing a Receiver instance.
Uncomment parameters in the Process section of the config file if needed. The default parameters are suited for Q26.
