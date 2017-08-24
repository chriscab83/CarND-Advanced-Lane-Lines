import cv2
import glob
import pickle
import plotter
import numpy as np
import image_gen as ig
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from image_handler import ImageHandler


def draw_region(img, pts, color=[255, 0, 0], lwidth=5):
    img = np.copy(img)
    cv2.line(img, pts[0], pts[1], color, lwidth)
    cv2.line(img, pts[1], pts[2], color, lwidth)
    cv2.line(img, pts[2], pts[3], color, lwidth)
    cv2.line(img, pts[3], pts[0], color, lwidth)
    return img


# Test the combination binary made from the color and sobel binary
img = mpimg.imread('../test_images/test4.jpg')
color_bin = ig.color_binary(img)
sobel_bin = ig.sobel_binary(img)
img_stacked = np.dstack((np.zeros_like(sobel_bin), sobel_bin, color_bin)) * 255
img_binary = ig.binary_image(img)

plotter.plot_images([img, img_stacked, img_binary],
                    ['Original', 'Stacked Binary', 'Combined Binary'],
                    file_path='image_binary.jpg', show=False)

# Load the camera calibration
img_size = (img.shape[1], img.shape[0])
dist_pickle = pickle.load(open('../camera_cal/dist_pickle.p', 'rb'))
mtx = dist_pickle['mtx']
dist = dist_pickle['dist']
img_handler = ImageHandler(mtx, dist, img_size)

# Test image warping on straight lined images
images = glob.glob('../test_images/straight_lines*.jpg')
for idx, fname in enumerate(images):
    straight_img = mpimg.imread(fname)
    warped = img_handler.warp(straight_img)
    warped = draw_region(warped, [(300, 720), (300, 0), (img_size[0] - 300, 0), (img_size[0] - 300, 720)])
    unwarped = img_handler.unwarp(warped)
    plotter.plot_images([straight_img, warped, unwarped],
                        ['Original', 'Warped', 'Unwarped'],
                        file_path='straight_warped' + str(idx + 1) + '.jpg', show=False)

# Test image binary warping
images = glob.glob('../test_images/test*.jpg')
for idx, fname in enumerate(images):
    img = mpimg.imread(fname)
    bin = ig.binary_image(img)
    warped = img_handler.warp(bin)
    plotter.plot_images([img, bin, warped],
                        ['Original', 'Thresh', 'Warped'],
                        file_path='thresh_warped' + str(idx + 1) + '.jpg', show=False)

# Find lane line finding starting point using histogram
img = mpimg.imread('../test_images/test5.jpg')
bin = ig.binary_image(img)
warped = img_handler.warp(bin)
histogram = np.sum(warped[warped.shape[0] // 2:, :] / 255, axis=0)

f, ax = plt.subplots(figsize=(50, 10))
x = range(350)
ax.imshow(warped, extent=[0, img.shape[1], 0, 700], cmap='gray')
ax.plot(histogram, color='firebrick', linewidth=2.0)
f.savefig('./writeup_imgs/histogram.jpg')
plt.close()

# Test sliding window lane finding
out_img = np.array(cv2.merge((warped, warped, warped)), np.uint8)

midpoint = np.int(histogram.shape[0] / 2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint

nwindows = 10
window_height = np.int(warped.shape[0] / nwindows)

nonzero = warped.nonzero()
nonzerox = np.array(nonzero[1])
nonzeroy = np.array(nonzero[0])

cur_leftx = leftx_base
cur_rightx = rightx_base

margin = 100
minpix = 250

left_lane_inds = []
right_lane_inds = []

for window in range(nwindows):
    y_low = warped.shape[0] - (window + 1) * window_height
    y_high = warped.shape[0] - window * window_height
    xleft_low = cur_leftx - margin
    xleft_high = cur_leftx + margin
    xright_low = cur_rightx - margin
    xright_high = cur_rightx + margin

    cv2.rectangle(out_img, (xleft_low, y_low), (xleft_high, y_high), (0, 255, 0), 2)
    cv2.rectangle(out_img, (xright_low, y_low), (xright_high, y_high), (0, 255, 0), 2)

    left_inds = ((nonzeroy >= y_low)
                 & (nonzeroy < y_high)
                 & (nonzerox >= xleft_low)
                 & (nonzerox < xleft_high)).nonzero()[0]

    right_inds = ((nonzeroy >= y_low)
                  & (nonzeroy < y_high)
                  & (nonzerox >= xright_low)
                  & (nonzerox < xright_high)).nonzero()[0]

    left_lane_inds.append(left_inds)
    right_lane_inds.append(right_inds)

    if len(left_inds) > minpix:
        cur_leftx = np.int(np.mean(nonzerox[left_inds]))

    if len(right_inds) > minpix:
        cur_rightx = np.int(np.mean(nonzerox[right_inds]))

left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)

leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds]
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]

left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
left_fitx = np.array(left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2], np.int32)
right_fitx = np.array(right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2], np.int32)

out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

f, ax = plt.subplots(figsize=(20, 10))
ax.imshow(out_img, cmap='gray')
ax.plot(left_fitx, ploty, color='yellow')
ax.plot(right_fitx, ploty, color='yellow')
ax.set_title('Sliding Window', fontsize=30)
f.savefig('./writeup_imgs/sliding_window.jpg')
plt.close()

# use found center points to draw lane
ym_per_pix = 30./720
xm_per_pix = 3.7/700

left_line = np.array(list(zip(np.concatenate((left_fitx-margin/4, left_fitx[::-1]+margin/4), axis=0),
                              np.concatenate((ploty, ploty[::-1]), axis=0))), np.int32)

right_line = np.array(list(zip(np.concatenate((right_fitx-margin/4, right_fitx[::-1]+margin/4), axis=0),
                               np.concatenate((ploty, ploty[::-1]), axis=0))), np.int32)

lane = np.array(list(zip(np.concatenate((left_fitx+margin/4, right_fitx[::-1]-margin/4), axis=0),
                         np.concatenate((ploty, ploty[::-1]), axis=0))), np.int32)

road = np.zeros_like(img)
cv2.fillPoly(road, [left_line], color=[255, 0, 0])
cv2.fillPoly(road, [right_line], color=[0, 0, 255])
cv2.fillPoly(road, [lane], color=[0, 255, 0])
road = img_handler.unwarp(road)

result = cv2.addWeighted(img, 1.0, road, 0.25, 0.0)

#curve_fit_cr = np.polyfit(np.array(ploty,np.float32)*ym_per_pix, np.array(leftx,np.float32)*xm_per_pix, 2)
#curvead = ((1 + (2*curve_fit_cr[0]*ploty[-1]*ym_per_pix + curve_fit_cr[1])**2)**1.5) / np.absolute(2*curve_fit_cr[0])

camera_center = (left_fitx[-1] + right_fitx[-1])/2
center_diff = (camera_center-warped.shape[1]/2)*xm_per_pix
side_pos = 'left'
if center_diff <= 0:
    side_pos = 'right'

#cv2.putText(result, 'Radius of Curvature = '+str(round(curvead,3))+'(m)', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(result, 'Vehicle is '+str(abs(round(center_diff, 3)))+'m '+side_pos+' of center', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

plotter.plot_images([result], ['Result'], file_path='result.jpg', show=False)


# turn into pipeline and run all test images
def pipeline(img):
    out_img = np.array(cv2.merge((warped, warped, warped)), np.uint8)

    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 10
    window_height = np.int(warped.shape[0] / nwindows)

    nonzero = warped.nonzero()
    nonzerox = np.array(nonzero[1])
    nonzeroy = np.array(nonzero[0])

    cur_leftx = leftx_base
    cur_rightx = rightx_base

    margin = 100
    minpix = 250

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        y_low = warped.shape[0] - (window + 1) * window_height
        y_high = warped.shape[0] - window * window_height
        xleft_low = cur_leftx - margin
        xleft_high = cur_leftx + margin
        xright_low = cur_rightx - margin
        xright_high = cur_rightx + margin

        cv2.rectangle(out_img, (xleft_low, y_low), (xleft_high, y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (xright_low, y_low), (xright_high, y_high), (0, 255, 0), 2)

        left_inds = ((nonzeroy >= y_low)
                     & (nonzeroy < y_high)
                     & (nonzerox >= xleft_low)
                     & (nonzerox < xleft_high)).nonzero()[0]

        right_inds = ((nonzeroy >= y_low)
                      & (nonzeroy < y_high)
                      & (nonzerox >= xright_low)
                      & (nonzerox < xright_high)).nonzero()[0]

        left_lane_inds.append(left_inds)
        right_lane_inds.append(right_inds)

        if len(left_inds) > minpix:
            cur_leftx = np.int(np.mean(nonzerox[left_inds]))

        if len(right_inds) > minpix:
            cur_rightx = np.int(np.mean(nonzerox[right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    left_fitx = np.array(left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2], np.int32)
    right_fitx = np.array(right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2], np.int32)

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    f, ax = plt.subplots(figsize=(20, 10))
    ax.imshow(out_img, cmap='gray')
    ax.plot(left_fitx, ploty, color='yellow')
    ax.plot(right_fitx, ploty, color='yellow')
    ax.set_title('Sliding Window', fontsize=30)
    f.savefig('./writeup_imgs/sliding_window.jpg')
    plt.close()

    # use found center points to draw lane
    ym_per_pix = 30. / 720
    xm_per_pix = 3.7 / 700

    left_line = np.array(list(zip(np.concatenate((left_fitx - margin / 4, left_fitx[::-1] + margin / 4), axis=0),
                                  np.concatenate((ploty, ploty[::-1]), axis=0))), np.int32)

    right_line = np.array(list(zip(np.concatenate((right_fitx - margin / 4, right_fitx[::-1] + margin / 4), axis=0),
                                   np.concatenate((ploty, ploty[::-1]), axis=0))), np.int32)

    lane = np.array(list(zip(np.concatenate((left_fitx + margin / 4, right_fitx[::-1] - margin / 4), axis=0),
                             np.concatenate((ploty, ploty[::-1]), axis=0))), np.int32)

    road = np.zeros_like(img)
    cv2.fillPoly(road, [left_line], color=[255, 0, 0])
    cv2.fillPoly(road, [right_line], color=[0, 0, 255])
    cv2.fillPoly(road, [lane], color=[0, 255, 0])
    road = img_handler.unwarp(road)

    result = cv2.addWeighted(img, 1.0, road, 0.25, 0.0)

    # curve_fit_cr = np.polyfit(np.array(ploty,np.float32)*ym_per_pix, np.array(leftx,np.float32)*xm_per_pix, 2)
    # curvead = ((1 + (2*curve_fit_cr[0]*ploty[-1]*ym_per_pix + curve_fit_cr[1])**2)**1.5) / np.absolute(2*curve_fit_cr[0])

    camera_center = (left_fitx[-1] + right_fitx[-1]) / 2
    center_diff = (camera_center - warped.shape[1] / 2) * xm_per_pix
    side_pos = 'left'
    if center_diff <= 0:
        side_pos = 'right'

    # cv2.putText(result, 'Radius of Curvature = '+str(round(curvead,3))+'(m)', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, 'Vehicle is ' + str(abs(round(center_diff, 3))) + 'm ' + side_pos + ' of center', (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return result


images = glob.glob('../test_images/test*.jpg')
for idx, fname in enumerate(images):
    img = mpimg.imread(fname)
    result = pipeline(img)
    plotter.plot_images([img, result],
                        ['Original', 'Result'],
                        file_path='result' + str(idx + 1) + '.jpg', show=False)
