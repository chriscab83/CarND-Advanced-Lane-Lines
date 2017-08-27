## Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./writeup_imgs/undist.jpg "Undistorted"
[image2]: ./writeup_imgs/undist_testimg.jpg "Undistorted Test Image"
[image3]: ./writeup_imgs/bin_r.jpg "RBG - R Channel"
[image4]: ./writeup_imgs/bin_v.jpg "HSV - V Channel"
[image5]: ./writeup_imgs/bin_b.jpg "LAB - B Channel"
[image6]: ./writeup_imgs/bin_color.jpg "Color Binary"
[image7]: ./writeup_imgs/bin_abs_sobel.jpg "Abs Sobel Binary"
[image8]: ./writeup_imgs/bin_mag.jpg "Magnitude Binary"
[image9]: ./writeup_imgs/bin_dir.jpg "Direction Binary"
[image10]: ./writeup_imgs/combined_sobel.jpg "Sobel Binary"
[image11]: ./writeup_imgs/bin_combined.jpg "Combined Binary"
[image12]: ./writeup_imgs/region_of_interest.jpg "Region"
[image13]: ./writeup_imgs/straight_warped1.jpg "Straight Line Sample 1"
[image14]: ./writeup_imgs/straight_warped2.jpg "Straight Line Sample 2"
[image15]: ./writeup_imgs/thresh_warped2.jpg "Threshold and Warped"
[image16]: ./writeup_imgs/histogram.jpg "Histogram"
[image17]: ./writeup_imgs/sliding_window.jpg "Sliding Window"
[image18]: ./vid_images/1503799569.647116.jpg "Smart Sliding Window"
[image19]: ./writeup_imgs/processed5.jpg "Pipeline Output"
[video1]: ./output_videos/project_video_out.1503799559.273496.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first and second code cell of the IPython notebook located in "./Advanced-Lane-Finding.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Once the camera calibration variables were obtained with OpenCV, the same method could be used to undistort our test images:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

To better identify lane lines in my images, I explored many color spaces to find which ones best brought out the lane lines while ignore other noise. In the end, I landed on using the R channel from the RGB color space, the V channel from the HSV color space, and the B channel from the LAB color space.  I then took those channels and applied a threshold to them to come up with three seperate binary representations.  I then combined these binary images into a single combined binary image.  Examples of the this process can be seen here:
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]

I then used a combination of sobel operator thresholds including absolute gradient, magnitude, and direction. Seen below:
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]

Finally, I combined teh sobel threshold binary and the color space binary into a single binary image to be used in my lane finding pipeline:
![alt text][image11]

I did much of my exploration of color space in the jupyter notebook found above as well as threshold discover and method testing.  Once I settled on the final threshold and methods to use in the pipeline, I moved the code to a python file and the code can be found in lines 23 through 124 in `image_processor.py` (src/image_processor.py).

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Again, for the perspective transform code, I did much of my testing in my jupyter notebook in code cells 14 - 17.  Once I found the settings that worked well for my perspective transform I moved the code to the ImageProcessor class mentioned above. The code for warping and unwarping methods can be found in that file in lines 15 - 21.  The ImageProcessor class takes input for the transform matrix and its inverse.  To get these variables, I decided to use hardcoded source and destination points as follows:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 265, 684      | 300, 720      | 
| 370, 610      | 300, 675      |
| 923, 610      | 980, 675      |
| 1035, 685     | 980, 720      |

This source region builds a short trapezoid in front of the vehicle. 
![alt text][image12]

I found that my transformation worked best by using this short trapezoid, which was easy to to find the parallel lines in, and mapping those points to a small section of the output image.  The results on the two straight line test images showed that the transformation was very successful and the lanes were parallel well into the distance passed the trapezoidal space chosen as my source points.
![alt text][image13]
![alt text][image14]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Once I had the image processing pipeline completed, I applied this pipeline to the test images by taking undistorting an image, getting the combined threshold binary of the image and warping the image to get a good look at the lane to find the lane lines. This process can be found in the `video_generator.py` file in lines 65 - 68.
![alt text][image15]

This warped image was then passed into the LaneFitter class where I used it to find a histogram of the lower portion of the image and zero in on a starting point of each lane line. `lane_fitter.py` lines 66 - 70
![alt text][image16]

With these starting points, I began a search up the image to correctly identify left and right lane line pixels in the image by sliding windows with a fixed size up the image at set intervals and finding the mean of pixels within that window.  That mean would then be used as the center point of the next window.  All pixels that fell into the left and right windows were said to belong to the left and right lane lines respectively. This process can be found in `lane_fitter.py` in lines 72 - 110. Once the pixels for each lane were found, they were passed to our Lane class and used along with the `cv2.polyfit()` method to find the polynomial fit to the pixels. `lane.py` line 37
![alt text][image17]

After the first frame and the LaneFitter finds a good reading for the left and right lanes, it then uses a smarter method of finding lane lines by using the best fit polynomials from the previous frame as the center points of and finding pixels up the polynomial in the window range to use as the lane pixels for the current frame. `lane_fitter.py` lines 38 - 63
![alt text][image18]

Also, to smooth the transitions in the polynomial from frame to frame, the Lane class applies averaging to the polynomials found over the past n frames. `lane.py` lines 46 - 47. I opted to use a weighted average because I found a straight average was causing a slow convergence on the actual lane when the vehicle shifts drastically between frames and the weighted average provided a smooth, but more accurate transition. 

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated the radius of the curvature using the formula provided.  This code can be found in `video_generator.py` on lines 105 and 106.  I used a value of 30 meters per 720 pixels on the y axis and 3.7 meters on the x axis as my transition variables to scale from screen space to real space. 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented a draw_lane_lines method on in lines 81 through 120 in my code in `video_generator.py`. This function takes in the image from a frame in the video, the left and right lane line polynomial fit, and the variables used for the radius of curvature calculation.  The result of the entire pipeline on a test image is as follows:
![alt text][image19]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/project_video_out.1503799559.273496.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I really enjoyed this project.  After exploring the binary space and the various settings for my window search, my pipeline worked very well on the project video.  However, as I worked on various other methods to make a more solid pipeline that would work on the challenge video, I found that most of the things I attempted were actually resulting in worse results in all videos, even the one I had working already.  Given more time, I would like to attempt implementing more sanity checks in my Lane class as well as my LaneFitter class which I believe would help my algorithm perform better in harder conditions.  My pipeline had a very hard time with the challenge video with the dark line down the middle of the road as well as the line created by the barrier.  It also did very poor in the lighting conditions going under the bridge.  In the warped screen at the very beginning of the video my algorithm completely lost the lane line and ended up converging on the left lane barrier as it was the only "lane" showing up in the top down image which lead to algorithm following that line for the majority of the video. 
![alt text][image20]

I also noticed the pipeline worked VERY poorly on the harder_challenge_video. This was clearly due to the sharper turns, hills, and short road segments. In these situations, I believe the algorithm would benefit greatly from using a smaller region of interest to better handle the hills and shorter road segments as well as increasing the margin the sliding windows were allowed to search to better find the sharper turns. I believe a smarter algorithm that could choose different look ahead distances given the quality of the top down image as well as the length of road visible in the image provided by the frame would greatly improve performance on both challenge videos.
