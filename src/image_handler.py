import numpy as np
import cv2


class ImageHandler:

    def __init__(self, mtx, dist, img_size):
        self.mtx = mtx
        self.dist = dist
        self.img_size = img_size

        top_w = 0.1
        bot_w = 0.87
        height = 0.37

        top_y = img_size[1]*(1.0-height)
        bot_y = img_size[1]
        top_rgt_x = (img_size[0]/2 + img_size[0]*top_w/2)
        top_lft_x = (img_size[0]/2 - img_size[0]*top_w/2)
        bot_rgt_x = (img_size[0]/2 + img_size[0]*bot_w/2)
        bot_lft_x = (img_size[0]/2 - img_size[0]*bot_w/2)

        pt1 = [top_lft_x, top_y]
        pt2 = [top_rgt_x, top_y]
        pt3 = [bot_rgt_x, bot_y]
        pt4 = [bot_lft_x, bot_y]

        pts = [pt1, pt2, pt3, pt4]

        src = np.float32(pts)

        offset = 200
        dst = np.float32([
            (offset, 0),
            (img_size[0]-offset, 0),
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
