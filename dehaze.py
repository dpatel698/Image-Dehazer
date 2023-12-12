# Darshil Patel
# September 24, 2022
# CS7180 Advanced Perception
# This script contains code for removing haze from an image using the dark channel prior
import cv2
from cv2.ximgproc import guidedFilter
import numpy as np


def get_dark_channel(b, g, r, kernel_ratio):
    """
    Computes the filtered dark channel image (as described in Section 3, Equation 5)

    :param b: blue channel
    :param g: green channel
    :param r: red channel
    :param kernel_ratio: ratio to determine the size of the minimum filter
    :return: the dark channel image
    """
    rows, cols = b.shape
    dark_channel = cv2.min(cv2.min(r, g), b)
    kernel_size = int(max(max(rows * kernel_ratio, cols * kernel_ratio), 3.0))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size), (-1, -1))
    dark_channel = cv2.erode(dark_channel, kernel)
    return dark_channel


def apply_dehazing(image, kernel_ratio, min_alight, eps):
    """
    Apply the dehazing algorithm to the image. The algorithm takes each rgb channel, obtains the dark channel
    prior, computes the transmission map and minimum atmospheric light, then applies a final transformation
    on each channel to dehaze the image.

    :param image: bgr image
    :param kernel_ratio: the ratio used to determine the local patch size
    :param min_alight: minimum atmospheric light
    :param eps: desired accuracy or change in parameters at which the iterative algorithm stops

    :return: dehazed image
    """
    image = image.astype("float32")
    # Split the bgr image into separate channels
    b, g, r = cv2.split(image)
    rows, cols = b.shape
    # Obtain dark channel (lowest intensity pixel map)
    dark_channel = get_dark_channel(b, g, r, kernel_ratio)
    # Transmission Map (Section 4.1)
    kernel_size = int(max(max(rows * kernel_ratio, cols * kernel_ratio), 3.0))
    transmission = dark_channel.copy()
    transmission = np.float32(transmission)
    transmission = transmission - 255.0
    transmission = transmission * -1.0
    transmission = transmission / 255.0
    gray = np.zeros((rows, cols))
    gray = np.float32(gray)
    cv2.cvtColor(image, cv2.COLOR_BGR2GRAY, gray)
    gray = gray / 255.0
    transmission = guidedFilter(gray, transmission, kernel_size * 5, eps)

    # Calculate the minimum atmospheric light from the dark channel (Section 4.3)
    minimum_atmospheric_light = min(min_alight, np.max(dark_channel))

    # Dehaze the separate channels and then merge for the final dehazed output image
    r_dehazed = dehaze_channel(r, transmission, minimum_atmospheric_light)
    g_dehazed = dehaze_channel(g, transmission, minimum_atmospheric_light)
    b_dehazed = dehaze_channel(b, transmission, minimum_atmospheric_light)
    dehazed_image = cv2.merge((b_dehazed, g_dehazed, r_dehazed))
    out = cv2.normalize(dehazed_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    dark_channel = cv2.normalize(transmission, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    dark_channel = cv2.normalize(dark_channel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return out, transmission, dark_channel


def dehaze_channel(channel, transmission, min_alight):
    """
    Dehaze the channel using the formula for recovering scence radiance equation 4.22
    :param channel: the channel to dehaze
    :param transmission: transmission image
    :param min_alight: minimum atmospheric light
    :return:
    """
    dehazed_channel = ((channel - min_alight) / transmission) + min_alight
    return dehazed_channel
