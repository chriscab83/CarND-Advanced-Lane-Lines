import cv2
import numpy as np
import image_gen as ig
from image_handler import ImageHandler

def pipeline(img, image_handler):

    # 1. get binary image
    binary_img = binary_image(img)

    # 2. warp binary
    warped = image_handler.warp(binary_img)

    # 3. get window centers


    # 4. poly fit


    # 5. draw


    # 6. unwarp


    # 7. return
    return img


def binary_image(img):
    color_bin = color_binary(img)
    sobel_bin = sobel_binary(img)
    binary_output = np.zeros_like(color_bin)
    binary_output[(sobel_bin == 1) | (color_bin == 1)] = 255
    return binary_output


def color_binary(img):
    r_thresh = (225, 255)
    v_thresh = (225, 255)
    b_thresh = (150, 255)

    r = img[:, :, 0]
    r_bin = binary_thresh(r, r_thresh)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2]
    v_bin = binary_thresh(v, v_thresh)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    b = lab[:, :, 2]
    b_bin = binary_thresh(b, b_thresh)

    combined = np.zeros_like(r_bin)
    combined[(r_bin == 1) | (v_bin == 1) | (b_bin == 1)] = 1
    return combined


def sobel_binary(img):
    gradx_params = {'thresh': (20, 100), 'ksize': 3}
    grady_thresh = {'thresh': (200, 255), 'ksize': 3}
    mag_thresh = {'thresh': (30, 100), 'ksize': 3}
    dir_thresh = {'thresh': (0.7, 1.3), 'ksize': 3}

    gradx_bin = sobel_abs_grad(img, orient='x', sobel_kernel=gradx_params['ksize'], thresh=gradx_params['thresh'])
    grady_bin = sobel_abs_grad(img, orient='x', sobel_kernel=grady_thresh['ksize'], thresh=grady_thresh['thresh'])
    mag_bin = sobel_mag_binary(img, sobel_kernel=mag_thresh['ksize'], thresh=mag_thresh['thresh'])
    dir_bin = sobel_dir_binary(img, sobel_kernel=dir_thresh['ksize'], thresh=dir_thresh['thresh'])

    combined = np.zeros_like(gradx_bin)
    combined[((gradx_bin == 1) | (grady_bin == 1)) | ((mag_bin == 1) & (dir_bin == 1))] = 1
    return combined


def sobel_abs_grad(img, orient='x', sobel_kernel=15, thresh=(20, 100)):
    r = img[:, :, 0]
    abs_sobel = np.absolute(cv2.Sobel(r, cv2.CV_64F, orient == 'x', orient == 'y', ksize=sobel_kernel))
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = binary_thresh(scaled_sobel, thresh)
    return binary_output


def sobel_mag_binary(img, sobel_kernel=3, thresh=(30, 100)):
    r = img[:, :, 0]
    sobelx = cv2.Sobel(r, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(r, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(mag) / 255
    mag = (mag / scale_factor).astype(np.uint8)
    binary_output = binary_thresh(mag, thresh)
    return binary_output


def sobel_dir_binary(img, sobel_kernel=9, thresh=(0.7, 1.3)):
    r = img[:, :, 0]
    sobelx = cv2.Sobel(r, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(r, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = binary_thresh(absgraddir, thresh)
    return binary_output


def binary_thresh(img, thresh=(0, 255)):
    binary = np.zeros_like(img)
    binary[(img > thresh[0]) & (img <= thresh[1])] = 1
    return binary