The geometric self-calibration for All-Sky Imagers tool uses a YAML to set the relevant parameters.

# Structure

The YAML structure for the geometric self-calibration is shown below. 

- Camera:
  - camera_name
  - camera_data_dir
  - img_path_structure
  - transforms
    - apply_camera_mask

- Calibration:
  - mode
  - path_orb_observations
  - path_calib_results
  - last_timestamp
  - last_timestamp_validation
  - target_calibration
  - orb_types
  - orb_types_validation
  - save_orb_quality_indicators
  - sort_out_imgs_manually
  - filter_detected_orbs
  - min_rel_dist_mask_orb
  - compute_and_save_azimuth_elevation
  - ignore_outliers_above_percentile
  - center_detection
    - x_samples
    - max_rel_center_dev
    - number_iterations
  - moon_detection
    - number_days
    - sampling_time
    - thresholds
      - min_area
      - max_area
      - intensity_threshold
      - aspect_ratio_tolerance
      - circularity_threshold
  - sun_detection
    - number_days
    - sampling_time
    - thresholds
      - min_area
      - max_area
      - intensity_threshold
      - aspect_ratio_tolerance
      - circularity_threshold
  - moon_validation
    - sampling_time
    - number_days
  - sun_validation
    - sampling_time
    - number_days
  - ss_statistics
    - mean
    - std


# Fields description


## Camera


| Property Name | Type | Description |
|---------------|------|-------------|
| camera_name | string | Name of the camera to be calibrated (as used in camera_data yaml).|
| camera_data_dir | string | Path to the camera_data yamls. If this field is not present then the default folder 'asi-core/asi_core/camera_data' is used. |
| img_path_structure| string | Path to each image, containing {camera_name} where the camera name should be inserted, {timestamp:...} (e.g. {timestamp:%Y%m%d%H%M%S%f}) where the evaluated timestamp should be inserted and {exposure_time:d} for exposure time. |
| transforms.apply_camera_mask | boolean | Indicates wheather to apply camera mask or not. |

#### Example
```
Camera:
  camera_name: 'Sky_Cam_01'
  camera_data_dir: '/Local/calib_test/camera_data'
  img_path_structure: '/Sky_Images/{timestamp:%Y}/{camera_name}/{timestamp:%m}/{timestamp:%d}/{timestamp:%H}/{timestamp:%Y%m%d%H%M%S}_{exposure_time:d}.jpg'
  transforms:
    apply_camera_mask: True
```

## Calibration:  

The `Calibration` object is contains a list of mandatory parameters and a set of objects that are present or not depending on the type of calibration to be run.   

The list of mandatory fields in `Calibration` are the following:


| Property Name | Type | Description |
|---------------|------|-------------|
| mode | string | The running mode can be one of the following: 'calibrate_validate_from_images', 'validate_from_images', 'calibrate_from_csv' or 'validate_from_csv'|
| path_orb_observations | string | Path to csv with orb observations from first run. Only relevant if mode is 'calibrate_from_csv' |
| path_calib_results | string | Path to calibration results to be used in a validation |
| last_timestamp | timestamp | last timestamp to be included in the calibration |
| last_timestamp_validation | timestamp | last timestamp to be included in the calibration |
| target_calibration | string | The optimization target can be of the following: 'optimize_eor_ior_center', 'optimize_eor_ior' or 'optimize_eor'. |
| orb_types | list of string | Orb types used for calibration can be 'Sun' or 'Moon', either or both. |
| orb_types_validation | list of string | Orb types used for validation can be 'Sun' or 'Moon', either or both. |
| save_orb_quality_indicators | boolean | Indicates wheather to save orb quality indicators or not. |
| sort_out_imgs_manually | boolean | Indicates wheather or not images are manually sorted. |
| filter_detected_orbs | boolean | Indicates wheather to filter detected orbs or not. |
| min_rel_dist_mask_orb | float | 1.5 |
| compute_and_save_azimuth_elevation | boolean | Indicates wheather to compute and save azimuth and elevation or not. |
| ignore_outliers_above_percentile | float | Observations with deviation above corresponding percentile are excluded from calculation of deviation/ optimization |

#### Example
```
Calibration:
  mode: 'calibrate_validate_from_images'
  path_orb_observations: ''
  path_calib_results: ''
  last_timestamp: !!timestamp '2024-04-04T14:00:00+01:00'
  last_timestamp_validation: !!timestamp '2024-04-04T14:00:00+01:00'
  target_calibration: 'optimize_eor_ior_center'
  orb_types: ['Moon'] 
  orb_types_validation: ['Sun', 'Moon'] 
  save_orb_quality_indicators: True
  sort_out_imgs_manually: False
  filter_detected_orbs: True
  min_rel_dist_mask_orb: 1.5
  compute_and_save_azimuth_elevation: True
  ignore_outliers_above_percentile: 99 
```


### Calibration nested objects:  

#### center_detection

The field `center_detection` must be present when:
- `mode` is 'calibrate_from_images' and `target_calibration` is 'optimize_eor_ior' or 'optimize_eor_ior_center'.
- `mode` is 'calibrate_from_csv'.

It contains the following fields:

| Property Name | Type | Description |
|---------------|------|-------------|
| x_samples | int |  Number of samples on grid of potential center points. | 
| max_rel_center_dev| float |  Fraction of the image side length, maximum expected deviation of the lens center from the image center. | 
| number_iterations|  int  |  Number of iterations in each of which a grid of test points is evaluated. | 

#### Example
```
  center_detection:
    x_samples: 6
    max_rel_center_dev: 0.25
    number_iterations: 11
```

#### sun_detection and moon_detection

There are two possibilities `sun_detection` and `moon_detection` but both has the same structure.

The field `sun_detection` must be present when `mode` is 'calibrate_from_images' or 'validate_from_images' and 'Sun' is in `orb_types` or `orb_types_validation`.

The field `moon_detection` must be present when `mode` is 'calibrate_from_images' or 'validate_from_images' and 'Moon' is in `orb_types` or `orb_types_validation`.


| Property Name | Type | Description |
|---------------|------|-------------|
| number_days| int | Number of days prior to 'last_timestamp' to be included. |
| sampling_time| int | Minutes, use one sky image every 'sampling_time' minutes. |

The `sun_detection` and `moon_detection` has a thresholds field with the following structure:

| Property Name | Type | Description |
|---------------|------|-------------|
|min_area | int | Minimum area in pixels of valid orb observation; small enough to detect the orb but large enough not to detect other lights |
|max_area| int | Maximum area in pixels of valid orb observation; small enough to reject objects which would be unusually large for a moon observation |
|intensity_threshold|| 100 |
|aspect_ratio_tolerance|float| 0.2|
|circularity_threshold|float| 0.8|

Note: Threshold values from detection are used in validation.

#### Example
```
   moon_detection:
     number_days: 28
     sampling_time: 10
     thresholds:
       min_area: 10
       max_area: 10000
       intensity_threshold: 100
       aspect_ratio_tolerance: 0.2
       circularity_threshold: 0.8

   sun_detection:
     number_days: 30 
     sampling_time: 10 
     thresholds:
       min_area: 100
       max_area: 10000 
       intensity_threshold: 240
       aspect_ratio_tolerance: 0.2
       circularity_threshold: 0.6
```

#### sun_validation and moon_validation fields:

As with the detection, there are two possibilities `sun_validation` and `moon_validation` but both has the same structure.

The field `sun_validation` must be present when `mode` is 'validate_from_images' and 'Sun' is in `orb_types_validation`.

The field `moon_detection` must be present when `mode` is 'validate_from_images' and 'Moon' is in `orb_types_validation`.

Validation fields:
| Property Name | Type | Description |
|---------------|------|-------------|
| number_days | int | Number of days prior to `last_timestamp` to be included |
| sampling_time | int | Minutes, use one sky image every `sampling_time` minutes |


#### Example
```
   moon_detection:
     number_days: 28
     sampling_time: 10
    
   sun_detection:
     number_days: 30 
     sampling_time: 10 
```


#### ss_statistics

The field `ss_statistics` must be present when `mode` is 'calibrate_from_images', 'calibrate_from_csv' or 'validate_from_images'.

| Property Name | Type | Description |
|---------------|------|-------------|
| mean | python/tuple of floats | Mean values of ss. |
| std | python/tuple of floats | Std values of ss.|

#### Example
```
  ss_statistics:
    mean: !!python/tuple [-653.3, 2.677e-4, 4.498e-07]
    std: !!python/tuple [5.4, 2.89e-05, 3.066e-08]
```