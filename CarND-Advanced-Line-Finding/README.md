**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image (`birds-eye view`).
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration1_undist.jpg "Undistorted chess board"
[image2]: ./output_images/test5_undist.jpg "Distortion Corrected"
[image3]: ./output_images/solidWhiteRight_1280_720_undist.jpg "straight lines"
[image4]: ./output_images/solidWhiteRight_1280_720_check.jpg "verify camera perspective"
[image5]: ./output_images/test5_birdeye.jpg "Road Transformed"
[image6]: ./output_images/test5_binary.jpg "Binary Example"
[image7]: ./output_images/test5_warped.jpg "Warp Example"
[image8]: ./output_images/test5_debug.jpg "Fit Visual"
[image9]: ./output_images/test5_result.jpg "Output"
[video1]: ./project_result.mp4 "Video"

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the module file `calibration.py`. 

I start camera calibration in the method `calibrateCamera`. Here test images are read from an folder. They are sized to a predefined shape (720,1280). To reduce noise each image is median blurred using a kernel size of 5.
The cv2 camera calibration methods require gray shaded images, which is achieved `cv2.cvtColor()`

In `getChessPoints` I prepare for each image `object points`, which will be the (x, y, z) coordinates of the chessboard corners in the 3D space. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time `cv2.findChessboardCorners()` successfully detect all chessboard corners in an image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()`. 

The camera matrix, the distortion coefficients and the image shape are stored in a pickle dump file, to make it available for other modules. They can load the dictionary by calling function `getCalibration()`.

To show the result of the calibration, function `showCalibrateImage()` can be used. It displays a figure presenting the original and the undistorted image.

Finally in the main function of the calibration module all test images are read and calibrated using the found distortion correction using the `cv2.undistort()`. They are stored in the `output_images` folder.

One distortion corrected camera test image is shown below:
![alt text][image1]

###Pipeline (single images)
The pipeline is executed in the module `lineDetection.py` function `process_colored_image()`.
####1. Provide an example of a distortion-corrected image.
In the main function of module `lineDetection.py` the camera matrix, distortion coefficients and the used image shape are loaded. If required the test image is resized to this predefined shape.

In Line 106 again the image is distortion corrected calling  `cv2.undistort()`.

![alt text][image2]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of different color channels and thresholds to generate a binary image at line 112 in module `lineDetection.py` calling `binaryFromColorspaces()` in the module `imageProcessing.py`. 

Function `binaryFromColorspaces()` line 70 to line 77 blurs (kernel size 15) the images to reduce clutter and receives a binary mask from function `binaryFromRGB()` and another binary mask from function `binaryFromYUVGray()` in the same file. Both mask are overlayed using an **or** operator.

Function binaryFromRGB (line 58 to 67) create the binary mask using thresholds for each color channel of the image to extract white lines. The thresholds are selected by calculating the 99% quantile.

Function `binaryFromYUV()` (line 88 to 95) transform the image to the YUV color space calling `imageFromYUV()`. I use only the inverse of U chrominance,because it is very good on showing yellow coloured objects. The binary masked is created using a thresholds which is again the 99% quantile.

Combining these different binary masks leads to a good filter for for white and yellow lines.

Here's an example of my output for this step.

![alt text][image6]

I experimented as well using magnitude and direction gradient functions, but they did not improve the results. I found better results, concentrating on the approbiate color spaces and dynamic thresholds.

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code in module `transformPerspective.py` includes a function `transformPerspective()`in lines 12 through 14. It is called at line 110 in `lineDetection.py` and expects the transformation matrix M.

The transformation matrix has been calculated in the main function in `transformPerspective.py` lines 64 through 100.
It loades an image I selected from an video stream, because it shows a nearly perfect straight, flat road.

![alt text][image3].

On the undistorded image, I manually extracted image coordinates for the two lines and defined object coordinates.
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 235, 692      | 385, 720      | 
| 584, 448      | 385, 0        |
| 702, 448      | 930, 0        |
| 1080, 692     | 930, 720      |

I verified that my perspective transform was working as expected by drawing the `SRC` and `DST` points onto the test image and its warped counterpart to verify that the lines appear parallel in the warped image.For this I created function `showPerspectiveTransformation()` in `transformPerspective.py` lines 24 through 42.

![alt text][image4].

The calculated perspective matrix and the invers matrix are stored in a pickle file `perspective.p`.
They are loaded and returned in a dictionary by calling `getPerspective()` in module `transformPerspective.py`.

I decided to transform the perspective, before I build the binary mask. The results have been slightly better.

![alt text][image7].

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Lines are extracted in the module `line.py`. Entry function is `findLine()` from line 69 through 94.
The following steps are applied:
- function `findPeaks()` return a list of points that represent maximum brightness. Search area for the points is the averaged history line +-20 pixel in x direction. Peaks are found by identifying step wise over the y axis the maximum values in a histogram over the x axis. 
- calculate a second order polynom spline from the list of points.
- if the list of points contains only a few points, the `findPeaks()` is called again using a larger search area.
- again a second order polynom spline from the list of points is calculated.
- outliers are removed in the function `removeOutlier()`. Outliers are points that have a larger distance from the second order spline than 90% of all points. The numpy percentile function is very helpful for that.
- after the outliers are removed the final second order spline is calculated to get a better fit.
- the fit is added to a list representing the history of the line fits by calling `update`.

The example shows the found peaks and the splines. On the left line blue points are peaks, red a spline points.
On the right line blue again are peak, green are spline points.

![alt text][image8]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

For the radius of curvature I decided to create a center line as an average between the smoothed left and right line.
I did this in lines 76 through 85 in `lineDetection.py`. 

In the class Line (`line.py` lines 49 through 59, I implemented a function `calcCurveRad()`.
It creates a polyline of second order from the line points. The x/y coordinates of the points are multiplied by scaling factors, I measured using my camera perspective test image. Using the scaled points I calulated again a
second order polyline. Finally I calculated the radius using the well known formula. This function is called for the center at line 83 in the module `lineDetection.py`.

The relative position of the vehicle is calculated in function `calcOffset` in the module `line.py` lines 45 through 47. The difference between the image x center value and the x coordinate on the image bottom is multiplied by the scaling factor. The function is applied for the center line at line 84 in the module `lineDetection.py`.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 13 through 33 in my code in `lineDetection.py` in the function `drawLinesOnRoad()`.  Here is an example of my result on a test image:

![alt text][image9]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_result.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

As in the first project the main weekness is the sensitivity on the brightness of images and local shades from trees. This has improved in this project already. But there is still space to improve. 
If the binary mask are bad, there will be only few peaks found to determine a line. To get a solid polynom fit, correct line points are especially  needed in the upper area of the image. To improve that I would try to create the binary mask using local thresholds in the images. This should be more able to handle shades.

Looking on the harder_challange video, its obvious that image preprocessing must be much more adaptiv to different conditions and that these can change rapidly nearly from image to image.


