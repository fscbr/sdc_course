##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/hog_example.jpg
[image3]: ./output_images/96searchwindow.jpg
[image4]: ./output_images/sliding_window_testImages.jpg
[image4]: ./output_images/sliding_window_video.jpg
[image4]: ./output_images/sliding_window_heatmap.jpg
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./project_video.mp4

---
###Writeup / README

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is in module `trainSvm.py` function readDatabase().  
If True as parameter is handed over, only a randomized subset of samples is loaded and returned

As database for the training I used the GTI and KITTI datasets offered on the project page.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(3, 3)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

In module `trainSvm.py` I created a function `trainParamlist()` that loops over a list of parameter compinations. It uses the above described randomized subset of the database.

First I examined for the combination of colorspaces (RGB, HSV, HLS, YUV, LUV, YCrCb), the combination of color space channels, the feature spaces (spatial,histogram,hog) 
I gained maximum classification accuracy when using RGB or HSL,  spatial and  hog featurs, but no histogram features.

The results for the different combinations are shown in the table below:

|Color space| Spatial | Histogram | HOG | Accuracy |

In a second round I trained the Linear svm using only combinations of color channels. The best result I achieved using all channels, so I kept that.


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the class SGDClassifier. In the main function at `trainSvm.py` line 132 through 179. The already loaded car and notcar datasets and the train parameters are handed over to the function.

First the features are extracted calling the `extract_features()` function in module `featureDetection.py`
the two feature sets are stacked in a two line matrix calling vstack. a label column is added to the feature matrix(label 1 for cars and label 0 for notcars) in lines 157 through 165.

In line 162 to 164 the feature matrix is normalized to with 0 mean and unit variance using the class sclearn.prepocessing.StandardScaler.
The fit method calculates the mean and standard deviation, which is used for scaling. Calling transform, the data is scaled.

The feature matrix is devided in a randomized 80% train and 20% test dataset calling `train_test_split()`.

After the preprocessing the Classifier is created in line 173 and trained in line 174.

The function returns the scaler object, the trained classifier and the achieved accuracy.
As the scaler and the trained classifier are needed for future predictions a dictionary is created and stored to a pickle file named `svc.p`.

I choosed the SGDClassifier as it is fast and has more options to optimize. For this reason I created a function gridSearch, which allows to find the best setup for the classifier.
I defined the parameter space for the tuning by `[{'loss':["hinge","modified_huber","squared_hinge"],'alpha': [0.00001,0.0001,0.001,0.01],"penalty":["l1","l2","elasticnet"]}]`

The found optimal combination is loss function `hinge`,penalty=`elasticnet` and alpha =`0.0001`.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
The search windows are created in function `slideWindows()` in module `featureDetection.py` at line 140 through 195.

The fuction creates for an area of interest, for a defined scale and an overlap ratio as many windows as fit into the search area.
I expected to use larger scales for closer object (== higher y values) and smaller scales for more distant objects (== lower y values).
I choosed always quadratic windows, as the database images are as well quadratic and the perspective tranformation should be linear on both axis.

The overlap should not to be too big as its increases the amount of search windows and therefor computation costs.
I choosed as area of interest for y values between 390 and 660 to cover th elane area and to surpress the visible part of the car cockpit.

For the test images I used an area of interest from left to right as there is now history of detections available.
For the video processing, I choosed an area at the low left and low right as entry area for overtaking cars. I added a high center area for vehicles that are over taken.

The later described heat map is used to provide areas of interest where cars have been detected in previous images.



During the tests with the project video, I recognized that 


For the test images I created a grid of windows for y values above 390 to cover the lane area.
First I choosed 
I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  


