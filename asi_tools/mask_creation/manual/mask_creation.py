# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
A small tool to create ASI masks manually.
"""

import warnings

import cv2
import numpy as np
import os
import re
import pathlib
import scipy

import argparse

from asi_core.config import config_loader
from asi_core.camera import sky_imager, obstacle_mask


class ObstacleMaskDetection:
    """
    Handles the manual creation of an ASI mask to obscure obstacles in the ASI's field of view
    """
    def __init__(self, cfg):

        try:
            cv2.namedWindow()
        except Exception:
            warnings.warn('Apparently cv2 is installed headless. Please install the full opencv package to use this '
                          'interactive GUI tool!')

        self.max_intensity = cfg['max_intensity']
        self.params_cv_detection = cfg['cv_detection']
        self.image_pxl_size = cfg['image_pxl_size']
        self.image_path = cfg['img_path']

        self.save_name = re.split(r'\.jpeg|\.jpg|\.png', os.path.basename(self.image_path))[0]

        self.orig_img = cv2.imread(self.image_path)
        self.orig_img = obstacle_mask.adjust_gamma(self.orig_img, gamma=2.2)
        self.mask = np.zeros_like(self.orig_img[:, :, 0])

        self.gui_add_to_mask = [[]]
        self.gui_remove_from_mask = [[]]
        self.gui_previous_event = None
        
    def detect_mask_cv(self, params=None):
        """
        Applies computer vision methods to automatically detect a mask of obstacles obscuring the sky in the ASI image.

        :param params: Configuration parameters to the algorithm
        :return: automatically detected mask, dtype boolean, shape of greyscaled RGB input image
        """
        if params is None:
            params = self.params_cv_detection

        self.mask = obstacle_mask.detect_mask_cv(self.orig_img, self.max_intensity, params['gaussian_kernel'],
                                        params['adaptive_thres_block_size'], params['adaptive_thres_mean_offset'],
                                        params['erode_dilate_kernel'], params['margin_horizon'])

    def load_existing_mask(self, existing_mask_path):
        """
        Loads an existing mask from a .mat file
        :param params: Path to the mask .mat file
        """
        mask = sky_imager.load_camera_mask(existing_mask_path)
        self.mask = mask * self.max_intensity

    def click_and_crop(self, event, x, y, f, cb):
        """
        From user clicks polygons are created indicating image areas to be masked or not.
        """

        # if the left mouse button was clicked, record the starting (x, y)
        if event in [cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN]:
            image_copy = self.masked_img.copy()

        if event in [cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN]:
            if not event == self.gui_previous_event:
                self.gui_add_to_mask = [[]]
                self.gui_remove_from_mask = [[]]

            if event == cv2.EVENT_LBUTTONDOWN:
                self.gui_add_to_mask[-1].append([x, y])
                temp_poly = self.gui_add_to_mask
                color = (0, 255, 0)
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.gui_remove_from_mask[-1].append([x, y])
                temp_poly = self.gui_remove_from_mask
                color = (0, 0, 255)

            if len(temp_poly[-1]) == 1:
                print(temp_poly)
                cv2.circle(image_copy, temp_poly[0][0], radius=0, color=color, thickness=4)
            elif len(temp_poly[-1]) > 1:
                # draw a rectangle around the region of interest
                print(temp_poly)
                cv2.polylines(image_copy, np.array(temp_poly), True, color, 2)
            self.gui_previous_event = event

        elif event == cv2.EVENT_MBUTTONDOWN:
            self.apply_polygons()
            self.apply_mask()
            image_copy = self.masked_img.copy()
            cv2.imshow('Correct mask', self.masked_img)

        if event in [cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN]:
            cv2.imshow('Correct mask', image_copy)

    def apply_polygons(self):
        """
        Area inside polygon specified by the user is added to or removed from the mask.
        """
        if len(self.gui_remove_from_mask[-1]) > 1:
            temp_mask = np.stack((np.zeros(np.shape(self.mask)), self.mask, np.zeros(np.shape(self.mask))),
                                 axis=2).astype(np.uint8)
            cv2.fillPoly(temp_mask, np.array(self.gui_remove_from_mask), color=(0, 255, 0))
            self.mask = temp_mask[:, :, 1]
        if len(self.gui_add_to_mask[-1]) > 1:
            temp_mask = np.stack((np.zeros(np.shape(self.mask)), 255-self.mask, np.zeros(np.shape(self.mask))),
                                 axis=2).astype(np.uint8)
            cv2.fillPoly(temp_mask, np.array(self.gui_add_to_mask), color=(0, 255, 0))
            self.mask = 255-temp_mask[:, :, 1]

        self.gui_add_to_mask = [[]]
        self.gui_remove_from_mask = [[]]

    def refine_manually(self):
        """
        Lets user specify image regions to be added or removed from mask.
        """
        cv2.namedWindow('Correct mask', cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow('Correct mask', self.image_pxl_size, self.image_pxl_size)
        cv2.moveWindow('Correct mask', 20, 20)
        cv2.setMouseCallback('Correct mask', self.click_and_crop)
        cv2.setWindowTitle('Correct mask', 'Draw polygons with left mouse button to add to mask, right button to remove'
                                           ' from mask, middle button or "a"-key to finish a polygon, "c"-key to finish'
                                           ' mask refinement!')
        self.apply_mask()
        cv2.imshow('Correct mask', self.masked_img)

        while True:
            # display the image and wait for a keypress
            key = cv2.waitKey(1) & 0xFF
            # if the 'a' key is pressed, apply the polygons to mask and reset the polygons
            if key == ord('a'):
                self.apply_polygons()
                self.apply_mask()
                cv2.imshow('Correct mask', self.masked_img)
            # if the 'c' key is pressed, break from the loop
            if key == ord("c"):
                break
        # close all open windows
        cv2.destroyAllWindows()

    def apply_mask(self):
        """
        Applies mask to image used for mask creation
        """
        self.masked_img = self.orig_img + 40 * np.stack((np.zeros(np.shape(self.mask)), 1 - self.mask /
                                                         self.max_intensity, np.zeros(np.shape(self.mask))),
                                                        axis=2).astype(np.uint8)

    def save_mask_and_docu(self):
        """
        Saves the mask in legacy format and docu information

        The following is saved:
        - A mat file which contains the mask and the path to the image based on which it was created
        - A jpg image file visualizing the masked areas in the original image
        """
        self.apply_mask()
        cv2.imwrite('masked_' + self.save_name + '.jpg', self.masked_img)
        scipy.io.savemat('mask_' + self.save_name + '.mat', {'Mask': {'BW': self.mask.astype('bool'),
                                                                      'RawImage': self.image_path}})

