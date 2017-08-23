import numpy as np
import cv2


class ImageHandler:

    def __init__(self, mtx, dist, img_size):
        self.mtx = mtx
        self.dist = dist
        self.img_size = img_size

        offset_bot_x = 200
        offset_top_x = 40
        src = np.float32([
            (img_size[0]/2-offset_top_x, 445),
            (img_size[0]/2+offset_top_x, 445),
            (img_size[0]-offset_bot_x, 720),
            (offset_bot_x[0], 720)
        ])

        offset = 300
        dst = np.float32([
            (offset, -offset*1),
            (img_size[0]-offset, -offset*1),
            (img_size[0]-offset, img_size[1]),
            (offset, img_size[1])
        ])

        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)

    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def warp(self, img):
        undist = self.undistort(img)
        return cv2.warpPerspective(undist, self.M, self.img_size, flags=cv2.INTER_LINEAR)

    def unwarp(self, img):
        return cv2.warpPerspective(img, self.Minv, self.img_size, flags=cv2.INTER_LINEAR)

    def binary_thresh(img, thresh=(0, 255)):
        binary = np.zeros_like(img)
        binary[(img > thresh[0]) & (img <= thresh[1])] = 1
        return binary
