## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<img src="examples/example_output.jpg" alt="Lane line"/>

---

Overview
---

This project is an amalgamation of Python, OpenCV and my growing love for CV ;)

Pipeline used for lane line detection :
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Installations
---

1. Install Anaconda on your local setup (Choose Python 3 version - [Link](https://www.anaconda.com/distribution/))
2. Create an environment (More on environments [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html))
Open cmd and type - `> conda create --name myenv` This will create an environment with default python version, which is Python 3 for this project.
3. Activate the environment, using `> activate myenv`
4. Install Libraries :
OpenCV : `> conda install -c conda-forge opencv `
moviepy (used in this project to work with videos): `> pip install moviepy`

[image1]: ./output_images/distortion_correction_chessboard.png "Undistorted"
[image2]: ./output_images/distortion_correction_road.png "Road undistorted"
[image3]: ./output_images/X_gradient_and_color_thresholding.png "Thresholded"
[image4]: ./output_images/perspective_transform.png "Warp Example"
[image5]: ./output_images/histogram.png "Histogram"
[image6]: ./output_images/windows.png "Sliding window"
[image7]: ./output_images/Final_output.png "Final output"

---

### Undistoring images using save calibration parameters

##### Chessboard
![alt text][image1]

##### Road image
![alt text][image2]


### Thresholding - Color thresholding and Gradient thresholding
![alt text][image3]


### Perspective and Inverse perspective transform
![alt text][image4]


### Lane line identification

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

##### Final output image
![alt text][image7]
