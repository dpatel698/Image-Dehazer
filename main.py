# Main module for dehazing an image using the dark channel prior
# Based on:
# Single Image Haze Removal Using Dark Channel Prior
# By Kaiming He, Jian Sun, and Xiaoou Tang
# PDF for paper: https://projectsweb.cs.washington.edu/research/insects/CVPR2009/award/hazeremv_drkchnl.pdf

import sys
import cv2
from dehaze import apply_dehazing

if __name__ == '__main__':
    image_name = sys.argv[1]
    kernel_ratio, min_atmospheric_light, eps = .01, 240.0, 0.000001
    original_image = cv2.imread(image_name)
    dehazed_image, transmission, dark_channel = apply_dehazing(original_image, kernel_ratio, min_atmospheric_light, eps)
    cv2.imshow("Original Image", original_image)
    cv2.imshow("Transmission Image", transmission)
    cv2.imshow("Dark Channel Image", dark_channel)
    cv2.imshow("Dehazed Image", dehazed_image)
    cv2.waitKey(0)  # waits until a key is pressed
    cv2.destroyAllWindows()


