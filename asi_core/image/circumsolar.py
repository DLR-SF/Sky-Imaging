# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
This module provides functions to analyse ASI (All-Sky Imager) images.
"""

import numpy as np
import cv2

from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter, median_filter
from pvlib import solarposition
import logging

from asi_core.utils.geometry import spherical_to_cartesian, cartesian_to_spherical, to_radians


def get_sun_pos_in_asi_image(sun_ele, sun_az, ele_mat, az_mat):
    """
    Determines the pixel position of the sun in an ASI image.

    :param sun_ele: Sun elevation angle in degrees.
    :param sun_az: Sun azimuth angle in degrees.
    :param ele_mat: Elevation matrix of the ASI image.
    :param az_mat: Azimuth matrix of the ASI image.
    :return: (row, col) coordinates of the sun in the image.
    """
    sun_ele, sun_az = to_radians(sun_ele, sun_az)
    diff_mat_ele = np.abs(ele_mat-sun_ele)
    diff_mat_az = np.abs(az_mat-sun_az)
    diff_mat = diff_mat_ele + diff_mat_az
    sun_pos = np.argwhere(diff_mat == np.nanmin(diff_mat))[0]
    return sun_pos


def compute_sun_dist_map(sun_ele, sun_az, ele_mat, az_mat, apply_filter=False, size=5):
    """
    Computes the distance map from each pixel to the sun in an ASI image.

    :param sun_ele: Sun elevation angle in degrees.
    :param sun_az: Sun azimuth angle in degrees.
    :param ele_mat: Elevation matrix of the ASI image.
    :param az_mat: Azimuth matrix of the ASI image.
    :param apply_filter: Whether to apply median filtering (default: False).
    :param size: Kernel size for median filtering (default: 5).
    :return: Distance map (in degrees) to the sun.
    """
    sun_ele, sun_az = to_radians(sun_ele, sun_az)
    cart_sun = np.array(spherical_to_cartesian(sun_az, sun_ele, 1)).reshape(1, 3)
    az_mask = ~np.isnan(az_mat)
    cart_coord = np.array(spherical_to_cartesian(az_mat[az_mask], ele_mat[az_mask], 1)).T
    sun_dist_mat = np.nan*np.ones_like(az_mat)
    sun_dist_mat[az_mask] = np.rad2deg(np.arccos(-(cdist(cart_sun, cart_coord, 'cosine') -1)).reshape(-1))
    if apply_filter:
        sun_dist_mat = median_filter(sun_dist_mat, size=size)
    return sun_dist_mat


def compute_cloud_coverage_and_distance_to_sun(seg_mask, cam_mask, sun_dist_map, cloud_value=1):
    """
    Computes cloud coverage and the minimum distance between clouds and the sun.

    :param seg_mask: Segmentation mask of the sky (clouds vs. background).
    :param cam_mask: Camera mask indicating valid pixels.
    :param sun_dist_map: Distance map from each pixel to the sun.
    :param cloud_value: Value representing clouds in the segmentation mask (default: 1).
    :return: Tuple (cloud_coverage, min_dist_cloud, coord_cloud):
             - cloud_coverage: Fraction of the sky covered by clouds.
             - min_dist_cloud: Minimum distance between a cloud pixel and the sun.
             - coord_cloud: Coordinates of the closest cloud pixel to the sun.
    """
    is_cloud = seg_mask == cloud_value
    if is_cloud.sum() > 1:
        cloud_coverage = is_cloud.sum()/cam_mask.sum()
        min_dist_cloud = np.nanmin(sun_dist_map[is_cloud])
        coord_cloud = np.argwhere(sun_dist_map == min_dist_cloud)[0]
    else:
        cloud_coverage = 0.
        min_dist_cloud = np.inf
        coord_cloud = [0, 0]
    return cloud_coverage, min_dist_cloud, coord_cloud


def get_sun_dist(az, ele, timestamp, location):
    """
    Calculate the sun distance angle based on azimuth and elevation angles.

    :param az: Azimuth angles (in degrees) of the sun.
    :param ele: Elevation angles (in degrees) of the sun.
    :param timestamp: Specific timestamp for which the solar position is calculated.
    :param location: Dictionary containing the latitude, longitude, and altitude of the location.
    :return: Three values: sun distance angle, sun azimuth angle, and sun elevation angle.
    """

    # Check if datetime_obj is timezone aware
    if timestamp.tzinfo is None or timestamp.tzinfo.utcoffset(timestamp) is None:
        raise ValueError("be careful - timestamp must be timezone aware")

    # Calculate the sun's azimuth and altitude angles
    sun_pos = solarposition.get_solarposition(timestamp, location['lat'], location['lon'], location['alt'])
    sun_ele = sun_pos['apparent_elevation'].iloc[0]
    sun_az = sun_pos['azimuth'].iloc[0]

    r = np.ones(np.size(az))
    x, y, z = spherical_to_cartesian(np.reshape(az, -1), np.reshape(ele, -1), r)

    rot = Rotation.from_euler('z', sun_az, degrees=True)
    ex1 = rot.apply(np.asarray([x, y, z]).T)

    rot = Rotation.from_euler('x', 90-sun_ele, degrees=True)
    ex1 = rot.apply(ex1)

    az_sun_normal_plane, ele_sun_normal_plane, r_sun_normal_plane = cartesian_to_spherical(ex1[:, 0], ex1[:, 1], ex1[:, 2])
    sun_dist = np.pi/2 - ele_sun_normal_plane
    sun_dist = np.rad2deg(np.reshape(sun_dist, np.shape(az)))

    return sun_dist, sun_az, sun_ele


def get_saturated_mask(img, saturation_limit=240, gray_scale=True, channel_dim=-1):
    if gray_scale:
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        saturation_mask = img >= saturation_limit
    else:
        intensity_sum = np.sum(img, axis=channel_dim)
        intensity_threshold = 3 * saturation_limit
        saturation_mask = intensity_sum >= intensity_threshold
    return saturation_mask
