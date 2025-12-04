# How to run the geometric self-calibration for All-Sky Imagers?

This section describes the steps to run the geometric camera self-calibration functionality in asi-core.

Make sure that the asi-core package is properly installed. 

To learn how to setup asi-core see [Getting started](getting_started.md)


## Preparation

Before running the geometric self-calibration it is necessary to configure parameters related to the camera being calibrated and to the calibration process itself.

This information is contained in 3 different files that are input to the calibration program.

The following sections explain how to prepare these 3 input files.


It is recommended to create a working folder in a suited location where the config files and calibration results are saved temporarily.

### Camera data yaml file

In asi-core, each camera sample is described by its own camera data yaml file. A camera data file template can be found in asi-core:

`data/camera_data/ASI_Template.yaml`

The camera data file template should be modified with the specific information of the camera being calibrated. The template contains comments describing the different fields to help configure the camera. Some relevant considerations are:
- The `camera_name` identifies the camera uniquely and shall match with the camera name used in the image folder etc. 
- The period between the dates `mounted` and `demounted` shall include the calibration period. 
- The `internal_calibration` parameters can be taken over from a camera of the same type, as an initial approximiation. When calibrating a sample of a completely new camera type, the polynomial coefficients `ss` taken from another camera type should be scaled linearly to the new camera's image resolution (i.e. `ss_new[i] = ss_old[i]*resolution_new/resolution_old`). For some camera types it might be required to perform the calibration according to Scaramuzza for one sample of that camera type to receive suited start values for cameras of that type. In this case, large deviations would be noticed after the calibration. For the 5 camera types tested so far, it was not necessary to adjust the start values. 
- The parameter `internal_calibration/diameter` can be added if needed. It indicates the diameter of the exposed image area. By default it is estimated as the smaller image side length. Only if the exposed image area's diameter is much larger than the smaller image side length (i.e. the exposed area is cut off notably at the image boundary), the diameter may be set in the config file.
- The specified width and height shall correspond to the actual width and height of the sky images.
- The center of the camera lens (`xc`, `yc`) will be determined by the calibration and not be used as start value by default.
- If possible each camera should be installed horizontally leveled, with the Sun in the upper image part at solar noon.
In that case set 

  `external_orientation: [0.,   3.14,   1.57]` (northern hemisphere) or

  `external_orientation: [0.,   3.14,   -1.57]` (southern hemisphere).

  Note: This is only a precaution. The calibration procedure should be able also to work with strongly deviating external orientations.
- Specify your camera's exposure times under `exposure_settings/exposure_times`:
  - day and night indicate the exposure times (as list) used during day- and nighttime respectively
  - The camera exposure times can differ between day and night. If taking image series, multiple exposure times can be specified but only images with the lowest listed exposure time are used for calibration. When calibrating with WDR / HDR images or images with variable exposure time, set [0] as exposure time. Multiple images of the same exposure time (if applicable) for the same timestamp are not allowed. 
- `exposure_settings/tolerance_timestamp` controls the accepted time deviation in seconds between requested and found image timestamp
- `camera_mask_file`: The path to the camera mask shall be specified.

It is recommended to create a camera_data subfolder in the working directory and store the camera data yaml file in there.


### Camera mask file

In addition to the camera data yaml file it is necessary to have a camera mask file which is apt for the calibration and validation periods.

To learn how to create a camera mask see [Mask creation](mask_creation.md)

The path to the camera mask file needs to be stated in the above-mentioned camera data yaml file.


### Calibration config file

There is a set of parameters that need to be established to run the calibration. All these parameters are compiled in a calibration config yaml file. A calibration config file template can be found in asi-core:

`asi_tools/calibration/self_calibration_cfg.yaml`


The config file template can be modified with the specific information of the calibration that is about to be done. The template contains comments describing the different fields to help configure the calibration. Relevant fields in the config file that shall be adapted are:
- camera_name indicates the name of the camera to calibrate and should match the one specified in the camera data yaml file.
- camera_data_dir should provide the path to the camera data folder described in the previous sections.
- img_path_structure specifies the template string for the image files path. It can contain any combination of the following placeholders:
    - {camera_name} for the camera name,
    - {exposure_time} for the image exposure time.
    - {timestamp:dt_format} for the image timestamp. dt_format is a combination of the following placeholders:
                            %Y year, %m month, %d day, %H hour, %M minute, %S second (e.g. {timestamp:%Y%m%d%H%M%S})
   
    An example: /a/path/to/image/{timestamp:%Y}/{camera_name}/{timestamp:%Y%m%d%H%M%S}_00{exposure_time}.jpg
- mode defines what task is performed _calibration_, _validation_ or both of them. Additionally, if orb positions can either be detected from images are be taken from a csv file which was created in advance. _calibrate_validate_from_images_ should be used as default. The following modes are available:     
      - calibrate_from_csv: Perform only the calibration using a csv file of orb observations. 
      - validate_from_csv: Perform only the validation using a csv file of orb observations. 
      - calibrate_validate_from_images: Perform calibration and validation receiving orb positions from image files.
      - validate_from_images: Perform only the validation receiving orb positions from image files.

- last_timestamp indicates the last timestamp included in the calibration period
- last_timestamp_validation indicates the last timestamp included in the validation period
- moon_detection/number_days sets the length of the calibration period in days for orb Moon
- sun_detection/number_days sets the length of the calibration period in days for orb Sun
- moon_validation/number_days sets the length of the validation period in days for orb Moon
- sun_validation/number_days sets the length of the validation period in days for orb Sun

The rest of the parameters usually remain unchanged, at least for Mobotix cameras.

- `Calibration/target_calibration: 'optimize_eor_ior_center'` is the default setting. In this mode external orientation, lens distortion and lens center coordinates will be determined.

The period included in the calibration should be long enough to have orb observations in a wide 
range of sky areas regarding azimuth and zenith angle. Note that depending on location and season this can sometimes be 
difficult. In the best case use one of the following:
- Moon positions from at least half a year between summer and winter solstice.
- Moon positions from at least one moon phase in winter.
- Moon positions from at least one moon phase in summer AND sun positions from one month in summer.

A sampling time of 10 minutes is usually enough to get all orb positions. Consider to use larger sampling time in order to save computing resources. When working in very cloudy conditions, reducing the sampling time can help to detect Sun and Moon in short cloud-free periods. 
The visualization received from the calibration will help to estimate 
if a sufficient number of orb positions distributed rather homogeneously over a wide range of azimuth and zenith angles has been detected.



## Run the calibration program

Open a terminal. If applicable, open the conda environment in which asi-core is installed. Run: 

    python <path_to_repository>/asi_tools/calibration


The calibration program has the following arguments:

  - '-c', '--config': Path to calibration config file. If not provided, a file 'self_calibration_cfg.yaml' is expected to be in the current working directory.
  - '--last_timestamp': Last timestamp included in calibration in ISO format including timezone. If not provided, the timestamp is read from the config file or if not found there, set to 6 days after the most recent full moon date.
  - '--mode': One of four modes:
      - calibrate_from_csv: Perform only the calibration using a csv file of orb observations. 
      - validate_from_csv: Perform only the validation using a csv file of orb observations. 
      - calibrate_validate_from_images: Perform calibration and validation receiving orb positions from image files.
      - validate_from_images: Perform only the validation receiving orb positions from image files.

      If not provided, the mode is read from the config file.



The calibration could take some time to run, depending on the data connection to the image storage location, 
the computing resources and the number of timestamps included in the calibration. The program creates a log file `geometric_calib_processed_<start date and time>.log` that is updated during the process. Check this file to see how the calibration progresses. 
You will see in the log file that the orb detection loops through all timestamps of the calibration. Subsequently, the progress of the iterative center detection is logged. Thereafter, the orb detection will loop through all timestamps included in the calibration.

## Calibration results

Once the calibration is completed, in the working directory, the following files are created:

| Name                                                       | Description                                                                                                                                          |
|------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| `calib_<camera_name>_<start>_<end>.yaml`                   | All results of the calibration.                                                      |
| `calib_<camera_name>_<start>_<end>.mat`                    | All results of the calibration as mat file for the use of legacy MATLAB tools                                                                                     |
| `moon_observations_<start>_<end>.csv`                      | Moon observations included in the validation or calibration. The observations can be used to reproduce the calibration/validation. | 
| `sun_<start>_<end>.csv`                                    | Sun observations included in the validation or calibration. The observations can be used to reproduce the calibration/validation.  | 
| `calibrated_observations_<start>_<end>.csv`                | All orb observations included in the calibration. In this file the final calibration parameters are applied.              | 
| `validation_observations_<start>_<end>.csv`                | All orb observations included in the validation. In this file the final calibration parameters are applied.               | 
| `calibrated_observations_<start>_<end>.png`                | Chart showing the detected and expected orb positions in the calibration                                                               |
| `validation_observations_<start>_<end>.png`                | Chart showing the detected and expected orb positions in the validation                                                                |
| `azimuth_matrix_<camera_name>_<timestamp_processed>.npy`   | A matrix of the azimuth angle viewed by each pixel                                                                                      |
| `elevation_matrix_<camera_name>_<timestamp_processed>.npy` | A matrix of the elevation angle viewed by each pixel                                                                                    |

`<start>`, `<end>` indicate the timestamps bounding the period included in the calibration.


When the calibration is completed, the camera data file shall be updated with the calibration results from `calib_<camera_name>_<start>_<end>.yaml`. 

It is also recommended to save the calibration results together with the config files used in an accessible location, identified with the date on which it was performed. E.g.:

 `../CloudCamCalibration/<camera_name>/<calibration_date>/<calibration_results_and_configs>`


### Example of Calibration results

This is an example of how to check the results of a calibration. 
In this case, Moon positions were used for the calibration and Sun positions for the validation.

#### Calibration with the Moon
The Moon observations included in the calibration can be seen in the following figure (`calibrated_observations_*.png`):

 ![Moon positions in calibration](media_self_calib/calibrated_observations_20230205000000_20230804000000.png)

In the case of a successful calibration most expected and detected moon positions, represented by the blue and red dots 
in the figure, coincide well.
Accordingly, you should see a low value of RMSE in the text box at the bottom. 
Usually a small number of outliers can be found. After filtering out 1% of the data points with the largest deviations, 
RMSE is typically less than 2 pixels (at 4-megapixel resolution). An RMSE larger than 4 pixels (at 4-megapixel 
resolution) indicates a rather low quality of the calibration. Only these 99% of the data points were included in the 
optimization of the camera model's parameters.

You should see a large number of visualized orb positions spread over a wide range of azimuth and 
elevation angles in one half of the hemisphere, as shown in the figure above. If this condition is not fulfilled, the 
calibration may be 
over-fitted to the sky region from which the observations were received.

#### Validation with the Sun
The Sun observations included in the validation can be seen in the following figure (`validation_observations_*.png`):

![Sun positions in validation](media_self_calib/validation_observations_20230807063000_20240202163000.png)

In this case, sun positions are used for the validation and were not included in the calibration. 
The test should verify that deviations between sun positions from astronomic 
expectation and image processing are small. Slightly larger deviations are possible if the validation 
interval includes a high fraction of turbid or cloudy situations in which the sun disk may still appear roundish while 
being disturbed by these influences. Usually you should receive an RMSD of around 3 pixels.

### Optional refinement of calibration results

If sun positions are used for the calibration, stronger deviations in the orb positions from image processing are 
possible. 
This is caused in particular by lens soiling, increased turbidity, clouds near the sun, cirrus clouds in general. 
When calibrating with sun positions and if the dataset is unfavourable, it might be necessary to manually filter out 
low quality images. 
To do so, run the calibration once more with the following adaptions in the config file:

````
Calibration:
	mode: calibrate_from_csv  
	sort_out_imgs_manually: True
	path_orb_observations: '<path_to_orb_observations_for_calibration>.csv'  # specify path to csv with orb observations from first run here
````

This will lead through a dialog which requests to delete invalid images from the subfolder used_imgs in the working 
directory (copied from the original image folder). 
Erased images will then be excluded from the calibration.

