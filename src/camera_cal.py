import numpy as np
import plotter
import pickle
import glob
import cv2

objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

obj_points = []
img_points = []

images = glob.glob('../camera_cal/calibration*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    if ret is True:
        obj_points.append(objp)
        img_points.append(corners)

img = cv2.imread('../camera_cal/calibration3.jpg')
img_size = (img.shape[1], img.shape[0])

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)

dst = cv2.undistort(img, mtx, dist, None, mtx)

plotter.plot_images([img, dst], ['Original', 'Undistorted'], 'undist.jpg')

dist_pickle = {'mtx': mtx, 'dist': dist}
pickle.dump(dist_pickle, open('../camera_cal/dist_pickle.p', 'wb'))

print('Camera calibration complete.')
