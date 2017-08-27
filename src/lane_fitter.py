import cv2
import numpy as np
from lane import Lane


class LaneFitter:
    def __init__(self, nwindows=9, window_margin=100, minpix=50, smoothing_factor=10):
        # number of windows to use when searching warped image
        self.nwindows = nwindows

        # margin the window can search to left and right of center point
        self.window_margin = window_margin

        # minimum number of pixels to quantify a good window
        self.minpix = minpix

        # number of past results to use when averaging to produce smooth transitions
        self.smoothing_factor = smoothing_factor

        # lane holders
        self.left_lane = Lane(smoothing_factor)
        self.right_lane = Lane(smoothing_factor)

        # frames processed
        self.frame_count = 0

    def fit_lane_lines(self, warped):
        out_img = np.array(cv2.merge((warped, warped, warped)), np.uint8)

        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        window_height = np.int(warped.shape[0] / self.nwindows)
        left_idxs = []
        right_idxs = []

        if self.left_lane.best_fit is not None and self.right_lane.best_fit is not None:
            left_fit = self.left_lane.best_fit
            right_fit = self.right_lane.best_fit

            left_poly = left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2]
            left_idxs = (nonzerox > left_poly - self.window_margin) & (nonzerox < left_poly + self.window_margin)

            right_poly = right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2]
            right_idxs = (nonzerox > right_poly - self.window_margin) & (nonzerox < right_poly + self.window_margin)

            ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
            left_fitx = np.array(left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2], np.int32)
            right_fitx = np.array(right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2], np.int32)

            left_line = np.array(list(zip(
                np.concatenate((left_fitx - self.window_margin, left_fitx[::-1] + self.window_margin), axis=0),
                np.concatenate((ploty, ploty[::-1]), axis=0))), np.int32)

            right_line = np.array(list(zip(
                np.concatenate((right_fitx - self.window_margin, right_fitx[::-1] + self.window_margin), axis=0),
                np.concatenate((ploty, ploty[::-1]), axis=0))), np.int32)

            search_area = np.zeros_like(out_img)
            cv2.fillPoly(search_area, [left_line], color=[0, 255, 0])
            cv2.fillPoly(search_area, [right_line], color=[0, 255, 0])
            out_img = cv2.addWeighted(out_img, 1.0, search_area, 0.50, 0.0)

        else:
            histogram = np.sum(warped[-warped.shape[0] // int(self.nwindows/4):, :], axis=0)
            midpoint = np.int(histogram.shape[0] / 2)

            cur_leftx = np.argmax(histogram[:midpoint])
            cur_rightx = np.argmax(histogram[midpoint:]) + midpoint

            for window in range(self.nwindows):
                y_low = warped.shape[0] - (window + 1) * window_height
                y_high = warped.shape[0] - window * window_height
                xleft_low = cur_leftx - self.window_margin
                xleft_high = cur_leftx + self.window_margin
                xright_low = cur_rightx - self.window_margin
                xright_high = cur_rightx + self.window_margin

                cv2.rectangle(out_img, (xleft_low, y_low), (xleft_high, y_high), (0, 255, 0), 2)
                cv2.rectangle(out_img, (xright_low, y_low), (xright_high, y_high), (0, 255, 0), 2)

                good_left_idxs = ((nonzeroy >= y_low)
                                  & (nonzeroy < y_high)
                                  & (nonzerox >= xleft_low)
                                  & (nonzerox < xleft_high)).nonzero()[0]

                good_right_idxs = ((nonzeroy >= y_low)
                                   & (nonzeroy < y_high)
                                   & (nonzerox >= xright_low)
                                   & (nonzerox < xright_high)).nonzero()[0]

                left_idxs.append(good_left_idxs)
                right_idxs.append(good_right_idxs)

                if len(good_left_idxs) > self.minpix:
                    cur_leftx = np.int(np.mean(nonzerox[good_left_idxs]))

                if len(good_right_idxs) > self.minpix:
                    cur_rightx = np.int(np.mean(nonzerox[good_right_idxs]))

            # Concatenate the arrays of indices
            left_idxs = np.concatenate(left_idxs)
            right_idxs = np.concatenate(right_idxs)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_idxs]
        lefty = nonzeroy[left_idxs]
        rightx = nonzerox[right_idxs]
        righty = nonzeroy[right_idxs]

        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        left_fit = self.left_lane.update(leftx, lefty)
        right_fit = self.right_lane.update(rightx, righty)

        ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
        left_fitx = np.array(left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2], np.int32)
        right_fitx = np.array(right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2], np.int32)

        cv2.polylines(out_img, [np.array(list(zip(left_fitx, ploty)), np.int32)], False, (255, 255, 0), thickness=3)
        cv2.polylines(out_img, [np.array(list(zip(right_fitx, ploty)), np.int32)], False, (255, 255, 0), thickness=3)

        # left_fit = np.polyfit(lefty, leftx, 2)
        # right_fit = np.polyfit(righty, rightx, 2)

        self.frame_count += 1
        return left_fit, right_fit, out_img
