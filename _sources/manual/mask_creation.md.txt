# Camera Mask

When locating an  All Sky Imagers (ASI) it is common to have structures or other objects in the vicinity of the camera that remain within its field of view and therefore appear in the images. Having these objects in the images is often not desirable, and therefore masks need to be created to help discard these parts of the images.

This is an example of an image and its corresponding mask:

![Camera image](media_camera_mask/camera_image.jpg)

![Camera mask](media_camera_mask/camera_mask.png)


Masks are created for each camera given its particular location and are based on the (static) objects in the field of view of the camera at the time the mask is created. It should be kept in mind that the environment may evolve and the objects observed may change, making it necessary to update the mask.

Asi-core provides the functionality to automatically create camera masks simply by providing a selection of clear sky images. In addition to that, asi-core offers a GUI that allows the user to create masks or improve existing ones by manually drawing polygons which indicate areas to be added or removed from the camera mask.


## Preparation

Make sure that asi-core is properly installed. To learn how to setup asi-core see [Getting started](getting_started.md)

The manual mask creation option requires to install the opencv package **with head**. By default "opencv-python-headless" is installed with asi-core in order to save resources. 

## Automatic Mask Creation

The automatic mask creation program generates a mask identifying fixed obstructions in sky images, based on images from a clear day. It aggregates a given number of images from the provided list, computes an average image with optional grayscale conversion, histogram equalization, and blurring, and then generates a mask highlighting static objects present across the images. 

### Run automatic mask creation program

In a terminal, run: 

    python <path to repository>/asi_tools/mask_creation/auto

The mask creation program has the following arguments:

  - '--image_dir': Path to image directory. Images in subfolders are also considered.
  - '--mask_dir': Path to save calculated mask. If not provided, `./masks` subfolder is created in the working directory and mask results are saved there.
  - '--mask_name': Name of the camera mask file. Default is `mask_camera01_<date>`.
  - '--num_images': Number of images from the image directory used to compute the camera mask. Default value is 40.
  - '--image_stride': Sample every N-th image to span a wider time range (e.g., 100 means use every 100th image). Default values is 100.


### Automatic mask creation results

When the mask creation is succesfully complete, the new mask is saved as `.npy` file in the given mask directory. A mask is also saved as Matlab `.mat` file for the use of legacy tools. 
Results are also visualized as a binary mask overlaid on an image. The mask is applied by coloring masked regions with a specified color and blending it with the original image using alpha transparency. The resulting image shown is also saved in the given mask directory.

This is an example of an automatic mask creation result:

![Automatic Mask result](media_camera_mask/automatic_mask_result.jpg)

It is recommended to save mask files in the `camera_data` folder in a new sub folder `camera_masks` and then update the camera data config file (field `camera_mask_file`) with the relative path to the new camera mask.

## Manual Mask Creation 

The manual mask creation program allows the generation of camera masks by the user via a GUI where a camera image is displayed and the user can click polygons indicating areas to be masked or not. As a first step, Asi-core applies computer vision methods to the displayed image to create an initial mask approximation.

The manual mask creation can be used to create camera masks from scratch or in combination with the automatic mask creation described in the section above to improve the results produced.

### Config file

There is a set of parameters that need to be established to run the manual mask creation. All these parameters are compiled in a mask creation config `.yaml` file. A mask creation config file template can be found in asi_tools folder:

`<path to repository>/asi_tools/mask_creation/manual/mask_creation_cfg.yaml`

The config file template can be modified with the specific information of the mask creation that is about to be done. The template contains comments describing the different fields to help configure the mask creation. Some relevant fields in the config file that shall be updated are:

- image_pxl_size: Size of visualized images.
- img_path: Path to a recent image to be used as reference for the mask creation.
- do_load_existing_mask: True in case there is an existing mask to be updated. False, otherwise.
- existing_mask_path: Path to the existing mask.

The rest of the parameters usually remain unchanged. 

## Run manual mask creation program

In a terminal, run: 

    python <path to repository>/asi_tools/mask_creation/manual -c mask_creation_cfg.yaml

The mask creation program has the following arguments:

  - '-c': Path to mask creation config file. If not provided, a config file 'mask_creation_cfg.yaml' is expected in the working directory.


When running the manual creation program the following GUI is opened:

![Manual Mask GUI](media_camera_mask/manual_camera_mask.png)

#### Add mask area

Clicking with the left mouse button polygons are drawn in green as in the figure below. 

![Manual Mask GUI](media_camera_mask/manual_camera_mask_add.png)

By pressing the key `a` the polygon is added to the mask and displayed filled with green color as the one in the figure below.

![Manual Mask GUI](media_camera_mask/manual_camera_mask_add_result.png)

#### Remove mask area

Clicking with the right mouse button polygons are drawn in red as the one in figure below. 

![Manual Mask GUI](media_camera_mask/manual_camera_mask_remove.png)

By pressing the key `a` the polygon area is removed from the mask.

![Manual Mask GUI](media_camera_mask/manual_camera_mask_remove_result.png)


Once the mask is complete, by pressing the key `c` the GUI is closed and results are saved in the working directory. The mask is saved as a Matlab `.mat` files `mask_<image_name>.mat` and also the mask overlaid on the image  `masked_<image_name>.jpg` to check and document the result visually.

It is recommended to save mask files in the `camera_data` folder in a new sub folder `camera_masks` and then update the camera data file (field `camera_mask_file`) with the relative path to the new camera mask.


Note: Images are from Plataforma Solar de Almería PSA owned by CIEMAT.