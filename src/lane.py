import numpy as np


class Lane:
    def __init__(self, smoothing_factor=10):
        # did last check result in a successful read
        self.detected = False
        self.missing_cnt = 0

        # current radius of curvature
        self.radius_of_curvature = None

        # current distance from center of car
        self.dist_from_center = None

        # number of readings to average over
        self.smoothing_factor = smoothing_factor

        # last poly fit
        self.recent_fits = []

        # best poly fit over taken by averaging last smoothing_factor readings
        self.best_fit = None

        self.weights = []
        cur_weight = 1
        for i in range(smoothing_factor):
            self.weights.append(cur_weight)
            cur_weight /= 3.0

    def update(self, xs, ys):
        if len(xs) == 0 | len(ys) == 0:
            self.detected = False
            self.missing_cnt += 1
            return self.best_fit

        fit = np.polyfit(ys, xs, 2)
        if self.is_good_fit(fit) is False:
            self.detected = False
            self.missing_cnt += 1
            return self.best_fit

        self.recent_fits.append(fit)

        nfits = min(len(self.recent_fits), self.smoothing_factor)
        self.best_fit = np.average(self.recent_fits[-self.smoothing_factor:], weights=self.weights[-nfits:], axis=0)
        self.missing_cnt = 0
        return self.best_fit

    def is_good_fit(self, fit):
        if self.best_fit is None:
            return True

        diff = self.best_fit - fit
        if (abs(diff[0]) > 1) | (abs(diff[1]) > 5):  # | (abs(diff[2]) > 50):
            return False

        return True

    @staticmethod
    def get_curvature(image, fit, ym_per_pix=30. / 720, xm_per_pix=3.7 / 700):
        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        fitx = np.array(fit[0] * ploty ** 2 + fit[1] * ploty + fit[2], np.int32)

        curve_fit_cr = np.polyfit(np.array(ploty, np.float32) * ym_per_pix, np.array(fitx, np.float32) * xm_per_pix, 2)
        curvead = ((1 + (2 * curve_fit_cr[0] * ploty[-1] * ym_per_pix + curve_fit_cr[1]) ** 2) ** 1.5) / \
                  np.absolute(2 * curve_fit_cr[0])
        return curvead
