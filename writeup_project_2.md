## Writeup Template
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

[image1]: ./output_images/distortion_correction_chessboard.png "Undistorted"
[image2]: ./output_images/distortion_correction_road.png "Road undistorted"
[image3]: ./output_images/X_gradient_and_color_thresholding.png "Thresholded"
[image4]: ./output_images/perspective_transform.png "Warp Example"
[image5]: ./output_images/histogram.png "Histogram"
[image6]: ./output_images/windows.png "Sliding window"
[image7]: ./output_images/Final_output.png "Final output"


### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.
---

### Camera Calibration

The code for this step is contained in the second code cell of the IPython notebook located in "./Advanced Lane Finding.ipynb"

__Steps :__
* prepare object points `objp`, with the assumption that chessboard is flat in realworld and on x-y plane with z = 0
* prepare image points - these are calculated by using `findChessboardCorners()`
* these two array of points are stored in corresponding lists - `objpoints` and `imgpoints`
* `calibrateCamera()` using above two lists
* store camera matrix `mtx` and distortion coefficients `dist` in a file for later use

### Undistoring images using save calibration parameters

The code for this step is contained in the 4th code cell of the IPython notebook located in "./Advanced Lane Finding.ipynb"

__Steps :__
* Load saved camera matrix and distortion coefficients
* use `undistort()` to undistort given distorted image

##### Chessboard
![alt text][image1]


### Pipeline

#### 1. Thresholding - Color thresholding and Gradient thresholding

The code for this step is contained in the 7th code cell of the IPython notebook located in "./Advanced Lane Finding.ipynb"

__`img_threshold()`__ provides gradient thresholding and color thresholding and steps taken within this function are:

* __Step 1:__ Undistort imput image

##### Road image
![alt text][image2]

* __Step 2:__ Convert undistorted image to HSL color space - This is very importand step because it has been seen that L channel and S channel are more impervious to shadows and bright lights and ultimately gives better thresholded image.

* __Step 3:__ Apply sobel operator to L channel in x direction. Sobel gives -ve derivative when transitoin is from light pixels to dark pixels so absolute values are taken which just indicates strength of edge

* __Step 4:__ Thresholding the resultant image using uppar and lower threshold.

* __Step 5:__ Thresholding L channel using corresponding uppar and lower threshold.


![alt text][image3]

#### 2. Perspective and Inverse perspective transform

The code for this step is contained in the 9th code cell of the IPython notebook located in "./Advanced Lane Finding.ipynb"

Function `perspective_transform()` performs perspective transform given source and destination points.

I had taken an image into photoshop and found these coordinates for source and destination points.

```python
src = np.float32([[200, 720], 
                  [1100, 720], 
                  [595, 450], 
                  [685, 450]])
                  
dst = np.float32([[300, 720], 
                  [980, 720],
                  [300, 0], 
                  [980, 0]])
    
```
##### Perspective warped image
![alt text][image4]

#### 3. Lane line identification

All the functions under "Helper functions" are used to identify lane lines.

__`find_lane_pixels()`__

The function `find_lane_pixels()` used to calcuate both lanes x and y pixels that lie within a window. only for the first frame of the video this function is run. from the next frame onwards only pixels in the vicinity of the previously detected lane  are analysed to be part of lane more on this later.

if used to detect lane on only image then only `find_lane_pixels()` will be called by default which uses sliding window techique.

##### Steps taken to detect line using sliding window technique

__Hyperparameter for window:__
* nwindows = 9
* margin = 100
* minpix = 200

* __step 1:__ Get the histogram of warped input image

##### Histogram of above warped image
![alt text][image5]

* __step 2:__ find peaks in the histogram that serves as midpoint for our first window
* __step 3:__ choose hyperparameter for windows
* __step 4:__ Get x, y coordinates of all the non zero pixels in the image
* __step 5:__ for each window in number of windows get indices of all non zero pixels falling in that window
* __step 6:__ Get the x, y coordinates based off of these indices

##### Output of sliding window
![alt text][image6]

__`fit_poly()`__

This function fits the points on a 2nd order polynomial

Given x and y coordinates of lane pixels fir 2nd order polynomial through them
    
here the function is of y and not x that is
    
x = f(y) = Ay**2 + By + C
    
returns coefficients A, B, C for each lane (left lane and right lane)

__`search_around_poly()`__

This function is extension to function `find_lane_pixels()`.
    
From second frame onwards of the video this function will be run.

the idea is that we dont have to re-run window search for each and every frame.
    
once we know where the lanes are, we can make educated guess about the position of lanes in the consecutive frame,
    
because lane lines are continuous and dont change much from frame to frame(unless a very abruspt sharp turn).
    
This function takes in the fitted polynomial from previous frame defines a margin and looks for non zero pixels in that range only. it greatly increases the speed of the detection.

I have chose margin of 100 here.

```python

    # we have left fitted polynomia (left_fit) and right fitted polynomial (right_fit) from previous frame,
    # using these polynomial and y coordinates of non zero pixels from warped image, 
    # we calculate corrsponding x coordinate and check if lies within margin, if it does then
    # then we count that pixel as being one from the lane lines.
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin))).nonzero()[0]
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin))).nonzero()[0]
```

__`measure_curvature_real()`__

This function calculates curvature of the lane in the real world using conversion factor in x and y direction as :

    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
Formula for curavature `R = (1 + (2Ay + B)^2)^(3/2) / (|2A|)`
    
```pyhon
    # It was meentioned in one the course note that camera was mounted in the middle of the car,
    # so the postiion of the car is middle of the image, the we calculate the middle of lane using
    # two fitted polynomials
    car_pos = img_shape[1]/2
    left_lane_bottom_x = left_fit_cr[0]*img.shape[0]**2 + left_fit_cr[1]*img.shape[0] + left_fit_cr[2]
    right_lane_bottom_x = right_fit_cr[0]*img.shape[0]**2 + right_fit_cr[1]*img.shape[0] + right_fit_cr[2]
    lane_center_position = ((right_lane_bottom_x - left_lane_bottom_x) / 2) + left_lane_bottom_x
    car_center_offset = np.abs(car_pos - lane_center_position) * xm_per_pix
``` 

__`draw_lane()`__

Given warped lane image, undistorted image and lane cooefficients this function draws lane on undistorted image.

* first it generates y coordinates using

```python
ploty = np.linspace(0, undistorted_img.shape[0]-1, undistorted_img.shape[0])
```
* for each y value it calculates x values from the fitted polynomials for both the lanes 

* Plots these points onto a blanked image copied from warped image

* using inverse perspective transform these lanes are again mapped back real camera view

```python
newwarp = inv_perspective_transform(color_warp)
```

* Finally both the images are merged together

```python
result = cv2.addWeighted(undistorted_img, 1, newwarp, 0.3, 0)
```

### Finally curvature and vehical offset are written on output image

Here I used curvature values and offset values returned by `measure_curvature_real()`

```python
font = cv2.FONT_HERSHEY_SIMPLEX
fontColor = (0, 0, 0)
fontSize = 1
cv2.putText(final_img, 'Lane Curvature: {:.0f} m'.format(np.mean([measures[0],measures[1]])), 
                (500, 620), font, fontSize, fontColor, 2)
cv2.putText(final_img, 'Vehicle offset: {:.4f} m'.format(measures[2]), (500, 650), font, fontSize, fontColor, 2)
```

##### Final output image
![alt text][image7]

---

### Pipeline (video)

All the above described steps works seamlessly with input video. Each frame is processed by all the above functions to get final lane

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Where can my code break and how to improve it

* The code works flawlessly with videos like project_video which contains decent amount of challanges but drastic change in sunlight still baffles the code
* Very abruput sharp turn may cause problems as I am not looking at sliding windows each time.

__How can I improve this__
* For the first problem I need more fine tuning of thresholding maybe not just sobel but use laplacian method to detect edges.
* Implement a way in which if there seems no matching pixels to previous lane lines then again apply sliding window technique to freshly detect lane pixels

