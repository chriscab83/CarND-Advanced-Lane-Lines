import cv2
import time
import pickle
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from lane_fitter import LaneFitter
from image_processor import ImageProcessor
from moviepy.editor import VideoFileClip

# load sample image
img = mpimg.imread('../test_images/test5.jpg')

# get saved camera calibration values from pickle file
dist_pickle = pickle.load(open('../camera_cal/dist_pickle.p', 'rb'))
mtx = dist_pickle['mtx']
dist = dist_pickle['dist']


# define function to get transformation matrix and get matrix
def get_M(img, offset=300):
    img_size = (img.shape[1], img.shape[0])

    pts = [(265, 684), (370, 610), (923, 610), (1035, 685)]
    src = np.float32(pts)
    dst = np.float32(
        [(offset, img_size[1]),
         (offset, 675),
         (img_size[0] - offset, 675),
         (img_size[0] - offset, img_size[1]),
         ])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv


M, Minv = get_M(img)

# create variables to process images
ip = ImageProcessor(mtx, dist, M, Minv)
fitter = LaneFitter(nwindows=9, window_margin=75, minpix=150, smoothing_factor=3)


# define function to plot n images in grid
def plot_images(imgs, titles, file_path=None, show=True):
    n = len(imgs)

    f, axs = plt.subplots(1, n, figsize=(20, 10))
    for i in range(n):
        axs[i].imshow(imgs[i], cmap='gray')
        axs[i].set_title(titles[i], fontsize=30)

    if file_path is not None:
        f.savefig('../vid_images/' + file_path)

    if show is True:
        plt.show()

    f.tight_layout()
    plt.close()


# define process image pipeline for video feed
def process_image(img):
    bin_img = ip.comb_binary(img)
    undistort = ip.undistort(bin_img)
    warped = ip.warp(undistort)
    left_fit, right_fit, out_img = fitter.fit_lane_lines(warped)

    img = draw_lane_lines(img, left_fit, right_fit)

    if fitter.frame_count % 10 == 0:
        plot_images([img, warped, out_img],
                    ['Original', 'Warped', 'Windows'],
                    'challenge.' + str(time.time()) + '.jpg', show=False)
    return img


# draw lane on image
def draw_lane_lines(img, left_fit, right_fit, ym_per_pix=30. / 720, xm_per_pix=3.7 / 700):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = np.array(left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2], np.int32)
    right_fitx = np.array(right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2], np.int32)

    left_line = np.array(list(zip(
        np.concatenate((left_fitx - 10, left_fitx[::-1] + 10), axis=0),
        np.concatenate((ploty, ploty[::-1]), axis=0))), np.int32)

    right_line = np.array(list(zip(
        np.concatenate((right_fitx - 10, right_fitx[::-1] + 10), axis=0),
        np.concatenate((ploty, ploty[::-1]), axis=0))), np.int32)

    center_lane = np.array(list(zip(
        np.concatenate((left_fitx, right_fitx[::-1]), axis=0),
        np.concatenate((ploty, ploty[::-1]), axis=0))), np.int32)

    lane = np.zeros_like(img)
    cv2.fillPoly(lane, [left_line], color=[255, 0, 0])
    cv2.fillPoly(lane, [right_line], color=[0, 0, 255])
    cv2.fillPoly(lane, [center_lane], color=[0, 255, 0])
    lane = ip.unwarp(lane)
    result = cv2.addWeighted(img, 1.0, lane, 0.50, 0.0)

    curve_fit_cr = np.polyfit(np.array(ploty, np.float32) * ym_per_pix, np.array(left_fitx, np.float32) * xm_per_pix, 2)
    curvead = ((1 + (2 * curve_fit_cr[0] * ploty[-1] * ym_per_pix + curve_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * curve_fit_cr[0])

    camera_center = (left_fitx[-1] + right_fitx[-1]) / 2
    center_diff = (camera_center - img.shape[1] / 2) * xm_per_pix
    side_pos = 'left'
    if center_diff <= 0:
        side_pos = 'right'

    cv2.putText(result, 'Radius of Curvature = ' + str(round(curvead, 3)) + '(m)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)
    cv2.putText(result, 'Vehicle is ' + str(abs(round(center_diff, 3))) + 'm ' + side_pos + ' of center', (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return result

# process video
input_path = '../challenge_video.mp4'
output_path = '../output_videos/challenge_video_out.'+str(time.time())+'.mp4'

clip = VideoFileClip(input_path)
vid_clip = clip.fl_image(process_image)
vid_clip.write_videofile(output_path, audio=False)
