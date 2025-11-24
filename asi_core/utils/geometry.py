# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

import numpy as np
import cv2
import logging


def to_radians(elevation, azimuth):
    """
    Converts elevation and azimuth angles from degrees to radians.

    :param elevation: Elevation angle [º]
    :param azimuth: Azimuth angle [º], over positive x-axis, rotating around z-axis
    :return: (np.float64) Elevation and Azimuth angles in radians.
    """

    elevation_rad = np.deg2rad(elevation)

    if azimuth > 270:
        azimuth_rad = (np.pi / 2 - np.radians(azimuth)) % np.pi
    else:
        azimuth_rad = np.pi/2 - np.radians(azimuth)

    return elevation_rad, azimuth_rad


def spherical_to_cartesian(azimuth, elevation, r):
    """
    Transform spherical to cartesian coordinates

    :param azimuth: array of the azimuth angle [radians], over positive x-axis, rotating around z-axis
    :param elevation: array of the elevation angle [radians]
    :param r: array of the radius
    :return: arrays of the cartesian coordinates x, y, z (same unit as radius)
    """

    rcos_theta = r * np.cos(elevation)
    x = rcos_theta * np.cos(azimuth)
    y = rcos_theta * np.sin(azimuth)
    z = r * np.sin(elevation)

    return x, y, z


def cartesian_to_spherical(x, y, z):
    """
    Transform cartesian to spherical coordinates. See reverse function spherical_to_cartesian, for further convention.

    :param x: cartesian coordinate x
    :param y: cartesian coordinate y, same unit as x
    :param z: cartesian coordinate z, same unit as x
    :return: arrays of the azimuth angle [radians], elevation angle [radians], radius (same unit as x)
    """

    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    elevation = np.arctan2(z, hxy)
    azimuth = np.arctan2(y, x)

    return azimuth, elevation, r


def is_circle_contour(contour, aspect_ratio_tolerance=0.1, circularity_threshold=0.8):
    """
    Determines if a contour is circular based on its aspect ratio and circularity.

    :param contour: (numpy.ndarray) The contour to analyze.
    :param aspect_ratio_tolerance: (float, optional) The maximum difference between the aspect ratio of the contour's
        bounding rectangle and 1. Default is 0.1.
    :param circularity_threshold: (float, optional) The minimum circularity of the contour. Default is 0.7.

    :returns:
        - is_circular, True if the contour is circular, False otherwise
        - aspcet_ratio, Aspect ratio of the contour
        - circularity, Circularity of the contour
    """

    circularity = 0
    aspect_ratio = 1e10

    area = cv2.contourArea(contour)
    number_points_on_contour = np.shape(contour)[0]

    if (area > 0) and (number_points_on_contour > 4):
        _, (major_diameter, minor_diameter), _ = cv2.fitEllipse(contour)
        aspect_ratio = float(major_diameter) / minor_diameter
        perimeter = cv2.arcLength(contour, True)

        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter ** 2)

    logging.debug(f'Circle properties (requirement), aspect_ratio: {aspect_ratio} (<{1 + aspect_ratio_tolerance});'
                  f' circularity: {circularity} (>{circularity_threshold})')

    is_circular = abs(aspect_ratio - 1) < aspect_ratio_tolerance and circularity > circularity_threshold

    return is_circular, aspect_ratio, circularity
