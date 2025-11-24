asi_core
==============================

Overview
------------------------------
This repository provides a comprehensive set of tools for processing and analyzing all-sky images. It includes utilities for data acquisition, preprocessing, geometric calibration, and combining sky images with sensor data for advanced analysis and visualization. Designed for machine learning (ML) applications, atmospheric studies, and environmental monitoring, this toolkit streamlines the workflow from raw data to meaningful insights.

A detailed documentation can be found [here](https://dlr-sf.github.io/Sky-Imaging)

Features
------------------------------
✅ Data Acquisition:
- Recording, loading and managing all-sky images
- Handling metadata and timestamps
- Automated file organization

✅ Geometric Calibration:
- SuMo: Tool for automated geometric self-calibration

✅ Sky Image Preprocessing:
- Rescaling, grayscale conversion, and format handling
- Mask generation for removing obstructions (e.g., camera housings, objects)
- Undistortion of fish-eye view
- HDR image generation and exposure fusion

✅ Data Analysis & Visualization:
- Merging image and sensor data (e.g., meteorological measurements)
- Mapping timestamps and solar positions


Installation
------------------------------

For this repository python 3.10 or newer is required.
If using conda, first setup a new conda environment using the environment.yml. 
- For creating a new environment: 

      conda env create -f environment.yml

- For updating an existing one: 

      conda env update -f environment.yml

Afterward, install the required pip packages and *asi_core* itself by typing: 
  
    pip install -e . --no-build-isolation


Usage
------------------------------
tbd.


Project Organization
------------------------------

    ├── .gitattributes     <- Define list of LFS files.   
    ├── .gitignore         <- Define list of files to be ignored by git.   
    ├── .gitlab-ci.yml     <- Define jobs that are run during CI/CD.   
    ├── environment.yml    <- Define environment.   
    ├── LICENSE.txt        <- Licensed under the Apache 2.0 License. For details, see NOTICE.txt.
    ├── NOTICE.txt         <- Contains credits and attribution notices.
    ├── pyproject.toml     <- Makes project pip installable (pip install -e .) so src can be imported.
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── asi_core           <- Source code of asi_core package
    ├── asi_tools          <- Source code of asi_tools
    ├── data/camera_data   <- Camera and mask config files of DLR cameras.
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    ├── notebooks          <- Dedicated notebooks for DLR projects.
    ├── scripts            <- Dedicated scripts for DLR projects.
    ├── tests              <- Unit tests for asi_core package components
    └── validation         <- System tests for asi_core package components
