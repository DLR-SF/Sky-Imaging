# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/DLR-SF/sky_imaging/blob/main/NOTICE.txt

"""
This module provides functions to automatically generate camera masks for all-sky imagers.
"""

import numpy as np
import cv2
import scipy

from asi_core.image.image_loading import load_images


def save_mask(mask, mask_dir, mask_name):
    """
    Save a mask as both a `.npy` and a legacy `.mat` file.

    :param mask: The mask array to be saved.
    :type mask: numpy.ndarray
    :param mask_dir: The directory where the mask files will be saved.
    :type mask_dir: pathlib.Path
    :param mask_dir: The name of the mask files.
    :type mask_dir: string
    :returns: None
    """

    new_mask_file = mask_dir / f'{mask_name}.npy'
    np.save(new_mask_file, mask)
    
    new_mask_file_legacy = mask_dir / f'{mask_name}.mat'
    mask_struct = {'Mask': np.array([[[mask]]])}
    scipy.io.savemat(new_mask_file_legacy, mask_struct)
    

def adjust_gamma(image, gamma=1.0):
    """
    Apply gamma correction to an image by building a lookup table mapping the pixel values [0, 255] to their adjusted
    gamma values
    """
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")

    return cv2.LUT(image, table)


def compute_mask(avg_img):
    """
    Create an image mask based on the input image. Input image should be a daily/longterm average image based
    on equalized histogram to enhance contrast
    """

    # 1. Step: Convert image to Gray Scale
    gray_img = cv2.cvtColor(avg_img, cv2.COLOR_BGR2GRAY)

    # 2. Step: Conservative masking of dark pixels
    gray_img[gray_img < 10] = 0

    # 3. Step: Contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(3, 3))
    lab = cv2.cvtColor(
        avg_img, cv2.COLOR_BGR2LAB
    )  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    lab = cv2.merge((l2, a, b))  # merge channels
    new_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    clahe = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)  # convert from LAB to GRAY
    clahe[gray_img < 10] = 0

    # 4. Step: Increase contrast (gamma adjustment and histogram equalization)
    gamma_c = 1.3
    gamma = adjust_gamma(clahe, gamma=gamma_c)
    targetval = 120  # Choose a medium color value
    alpha = np.nanmin([1.8, targetval / np.nanmean(gamma)])
    beta = 0
    scaleabs = cv2.convertScaleAbs(gamma, alpha=alpha, beta=beta)

    # 5. Step: Gaussian Blurring to remove noises
    img = cv2.GaussianBlur(scaleabs, (3, 3), cv2.BORDER_DEFAULT)

    # 6. Step: Canny Edge detection
    cminval = 20
    cmaxval = 40
    edges = cv2.Canny(img, cminval, cmaxval, L2gradient=True)
    size = avg_img.shape[0]

    # 7. Step: Detect circular horizon as a next conservative masking
    circles = cv2.HoughCircles(
        scaleabs,
        cv2.HOUGH_GRADIENT,
        1,
        size / 2,
        param1=50,
        param2=20,
        minRadius=int(size / 2) - 100,
        maxRadius=int(size / 2) + 100,
    )

    if circles is not None:
        center = (int(circles[0, 0, 0]), int(circles[0, 0, 1]))
        radius = int(circles[0, 0, 2])
        # Subtract a margin of pixels to decrease the circle
        smaller_radius = radius - 20
        if np.abs(center[0] - size / 2) < 150:
            cv2.circle(edges, center, smaller_radius, 255, 1)
    else:
        print("No Hough circles detected")

    # 8. Step: Apply the maximum filter to strengthen the edges
    k = 5
    iter = 5
    kernel = np.ones((k, k), np.uint8)
    thres = cv2.dilate(edges, kernel, iterations=iter)

    # 9. Step: Smoothen the edges
    thres = cv2.medianBlur(thres, ksize=(2 * k) - 1)

    # 10. Step: Apply the Component analysis function
    analysis = cv2.connectedComponentsWithStats(cv2.bitwise_not(thres), 8, cv2.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis

    # 11. Step: Find the component corresponding to the inner area
    mask = np.zeros(thres.shape, dtype="uint8")
    xs, ys = int(mask.shape[0] / 2), int(mask.shape[1] / 2)
    lid = label_ids[xs, ys]
    componentMask = (label_ids == lid).astype("uint8") * 255
    # Final Step: Binary masking
    mask = cv2.bitwise_or(mask, componentMask)
    return mask


def aggregate_images(img_list, gray_scale=False, equalization=False, blur=False):
    """
    Process a list of images by applying optional grayscale conversion, histogram equalization,
    and blurring. Computes the average and standard deviation of the processed images.

    :param img_list: List of input images as NumPy arrays.
    :type img_list: list of numpy.ndarray
    :param gray_scale: Whether to convert images to grayscale (default is False).
    :type gray_scale: bool, optional
    :param equalization: Whether to apply histogram equalization to enhance contrast (default is False).
    :type equalization: bool, optional
    :param blur: Whether to apply a blurring filter to reduce noise (default is False).
    :type blur: bool, optional
    :return: A dictionary containing processed images, their average, and standard deviation.
    :rtype: dict
    """

    assert len(img_list) > 0, f'Empty list passed.'

    images = []
    for image in img_list:
        if gray_scale: image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if equalization:
            targetval = 120  # Choose a medium color value
            alpha = targetval / np.nanmean(image)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        if blur: image = cv2.blur(image, (3,3))
        images.append(image.astype('uint8')[None,...])
    images = np.vstack(images)

    return_dict = {
        'images': images,
        'avg_image': np.mean(images, axis=0).astype(np.uint8),
        'std_image': np.std(images, axis=0).astype(np.uint8)}

    return return_dict


def create_mask(img_files, num_images=-1):
    """
    Create a mask of static objects from a set of image files.

    This function aggregates a specified number of images from the provided list,
    computes an average image with optional grayscale conversion, histogram
    equalization, and blurring, and then generates a mask highlighting static
    objects present across the images.

    :param img_files: A list of image file paths to be used for mask generation.
    :type img_files: list[str] or list[pathlib.Path]
    :param num_images: The number of images to use for aggregation. If -1, use all provided images.
    :type num_images: int, optional
    :returns: The generated mask based on static objects detected in the aggregated image.
    :rtype: numpy.ndarray
    :raises ValueError: If `img_files` is empty or if `num_images` is 0.
    """
    img_list = load_images(img_files[0:num_images])
    agg_dict = aggregate_images(img_list, gray_scale=False, equalization=True, blur=True)
    mask = compute_mask(agg_dict['avg_image'])
    
    return mask


def detect_mask_cv(img, max_intensity, gaussian_kernel, adaptive_thres_block_size, adaptive_thres_mean_offset, 
                    erode_dilate_kernel, margin_horizon):
    """
    Applies computer vision methods to automatically detect a mask of obstacles obscuring the sky in the ASI image.

    :param img: Image to be used for mask generation.
    :param mask_intesity: 
    :param gaussian_kernel: 
    :param adaptive_thres_block_size: 
    :param adaptive_thres_mean_offset: 
    :param erode_dilate_kernel: 
    :param margin_horizon: 
    :return: automatically detected mask, dtype boolean, shape of greyscaled RGB input image
    """        

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to smooth out any noise
    blur = cv2.GaussianBlur(gray, gaussian_kernel, cv2.BORDER_CONSTANT)

    # Apply adaptive thresholding to binarize the image
    thresh = cv2.adaptiveThreshold(blur, max_intensity, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                    adaptive_thres_block_size, adaptive_thres_mean_offset)

    # Apply morphological operations to remove small objects and fill in gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, erode_dilate_kernel)
    erode = cv2.erode(thresh, kernel)
    dilate = cv2.dilate(erode, kernel)

    # Find contours and select the contour with the largest area
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sky_contour = max(contours, key=cv2.contourArea)

    # Create a mask from the selected contour
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [sky_contour], 0, max_intensity, -1)

    # Approximate the circle
    (center_x, center_y), radius = cv2.minEnclosingCircle(sky_contour)
    center = (int(center_x), int(center_y))
    radius = int(radius)

    # Create a new mask with the approximated circle
    circle_mask = np.zeros_like(gray)
    cv2.circle(circle_mask, center, radius, max_intensity, -1)

    # Subtract 16 pixels from the radius and create a smaller circle mask
    smaller_radius = radius - margin_horizon
    smaller_circle_mask = np.zeros_like(gray)
    cv2.circle(smaller_circle_mask, center, smaller_radius, max_intensity, -1)

    # Combine the original mask and the smaller circle mask
    mask = cv2.bitwise_and(mask, smaller_circle_mask)
    mask[gray < 5] = 0
    # Erode mask boundary by small margin
    kernel = np.ones(erode_dilate_kernel, np.uint8)

    mask = cv2.erode(mask, kernel, iterations=2)

    return mask