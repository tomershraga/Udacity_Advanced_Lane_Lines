## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./output_images/distortion_corrected_calibration_image.png "Undistorted"
[image2]: ./output_images/undistorted_on_test_image.png "Road Transformed"
[image3]: ./output_images/binary_images.png "Binary Example"
[image4]: ./output_images/lane_line.png "Fit Visual"
[image5]: ./output_images/final_result.png "Output"
[image6]: ./output_images/bird_eye_view.png "Bird Eye View"
[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for camera calibration can be found in `camera_calibration.py`.

The code for undistort the images can be found in `undistort.py`

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
I used the mtx and dist return values from camera calibration and used it on `cv2.undistort` function

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at function  create_thresholded_binary_image in `thresholded_binary.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `transform_perspective(threshold_image)`, which appears in lines 13 through 26 in the file `perspective_transform.py` .  The `transform_perspective(threshold_image)` function takes as inputs an image (`threshold_image`), and genetares source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.array([[200, 720], [1150, 720], [750, 480], [550, 480]], np.float32)
dst = np.array([[300, 720], [900, 720], [900, 0], [300, 0]], np.float32)
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 200, 720      | 300, 720      | 
| 1150, 720     | 900, 720      |
| 750, 480      | 900, 0        |
| 550, 480      | 300, 0        |

![alt text][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used the `find_lane_pixels` function in `find_lane.py`
Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this: I created an histogram of the lower half of the image to find the peak, this is the left and the right lanes. These peaks are considered as the starting point(x-positions) of the lane lines. From the peak, there are n number of sliding window with a margin of +/- 100 from starting points. I then, extracted the activated pixels(non-zero x and y points), within the window. These x and y activated pixels are fit in a polynomial using np.polyfit() to get the left and right coefficients respectively. With the coefficients and the y value of the image, x values of left and right lane are calculated using the second order polynomial equation.

![alt text][image4]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in `measure_curvature_pixels` function in `curvature_and_offset.py`.
I defined conversions in x and y from pixels space to meters `ym_per_pix = 30/720` and `xm_per_pix = 3.7/700`.
Then i defined y-value where we want radius of curvature and i choosed the maximum y-value, corresponding to the bottom of the image.
At the end i did a calculation of R_curve (radius of curvature) one for left and one for right.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in `original_lane.py` in the function `original_lane_lines(warp_img, undistorted_line_image, x_line_values, MatrInv)`.  Here is an example of my result on a test image:

![alt text][image5]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I had a problem with creating the out_img in search_around_poly function. It tooks me long time to understand that the input image was wrong i gave the original image instead the transformed perspective image.
