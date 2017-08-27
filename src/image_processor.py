import cv2
import numpy as np


class ImageProcessor:
    def __init__(self, mtx, dist, m, minv):
        self.mtx = mtx
        self.dist = dist
        self.M = m
        self.Minv = minv

    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def warp(self, img):
        img_size = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, self.M, img_size, flags=cv2.INTER_LINEAR)

    def unwarp(self, warped):
        img_size = (warped.shape[1], warped.shape[0])
        return cv2.warpPerspective(warped, self.Minv, img_size, flags=cv2.INTER_LINEAR)

    @staticmethod
    def binary_thresh(img, thresh=(0, 255)):
        binary = np.zeros_like(img)
        binary[(img > thresh[0]) & (img <= thresh[1])] = 1
        return binary

    def comb_binary(self, img):
        sobel_bin = self.sobel_binary(img)
        color_bin = self.color_binary(img)
        img_bin = np.zeros_like(sobel_bin)
        img_bin[(sobel_bin == 1) | (color_bin == 1)] = 255
        return img_bin

    def color_binary(self, img):
        r_thresh = (225, 255)
        v_thresh = (225, 255)
        b_thresh = (150, 255)

        r = img[:, :, 0]
        r_bin = self.binary_thresh(r, r_thresh)

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        v = hsv[:, :, 2]
        v_bin = self.binary_thresh(v, v_thresh)

        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        b = lab[:, :, 2]
        b_bin = self.binary_thresh(b, b_thresh)

        combined = np.zeros_like(r_bin)
        combined[(r_bin == 1) | (v_bin == 1) | (b_bin == 1)] = 1
        return combined

    def sobel_binary(self, img):
        gradx_params = {'thresh': (20, 100), 'ksize': 3}
        # grady_thresh = {'thresh': (200, 255), 'ksize': 3}
        mag_thresh = {'thresh': (30, 100), 'ksize': 3}
        dir_thresh = {'thresh': (0.9, 1.3), 'ksize': 3}

        gradx_bin = self.sobel_abs_grad_binary(img, orient='x', sobel_kernel=gradx_params['ksize'],
                                               thresh=gradx_params['thresh'])
        # grady_bin = self.sobel_abs_grad_binary(img, orient='x', sobel_kernel=grady_thresh['ksize'],
        #                                       thresh=grady_thresh['thresh'])
        mag_bin = self.sobel_mag_binary(img, sobel_kernel=mag_thresh['ksize'], thresh=mag_thresh['thresh'])
        dir_bin = self.sobel_dir_binary(img, sobel_kernel=dir_thresh['ksize'], thresh=dir_thresh['thresh'])

        combined = np.zeros_like(gradx_bin)
        combined[((gradx_bin == 1)) | ((mag_bin == 1) & (dir_bin == 1))] = 1
        return combined

    def sobel_abs_grad_binary(self, img, orient='x', sobel_kernel=15, thresh=(20, 100)):
        r = img[:, :, 0]

        # Take the derivative in x or y given orient = 'x' or 'y'
        # Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(cv2.Sobel(r, cv2.CV_64F, orient == 'x', orient == 'y', ksize=sobel_kernel))

        # Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        # Create a mask of 1's where the scaled gradient magnitude is between thresh
        binary_output = self.binary_thresh(scaled_sobel, thresh)

        # Return this mask as your binary_output image
        return binary_output

    def sobel_mag_binary(self, img, sobel_kernel=3, thresh=(30, 100)):
        r = img[:, :, 0]

        # Take the gradient in x and y separately
        sobelx = cv2.Sobel(r, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(r, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # Calculate the magnitude
        mag = np.sqrt(sobelx ** 2 + sobely ** 2)

        # Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scale_factor = np.max(mag) / 255
        mag = (mag / scale_factor).astype(np.uint8)

        # Create a binary mask where mag thresholds are met
        binary_output = self.binary_thresh(mag, thresh)

        # Return this mask as your binary_output image
        return binary_output

    def sobel_dir_binary(self, img, sobel_kernel=9, thresh=(0.9, 1.3)):
        r = img[:, :, 0]

        # Take the gradient in x and y separately
        sobelx = cv2.Sobel(r, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(r, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # Take the absolute value of the x and y gradients
        # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

        # Create a binary mask where direction thresholds are met
        # Return this mask as your binary_output image
        binary_output = self.binary_thresh(absgraddir, thresh)

        return binary_output
