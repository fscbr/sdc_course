##Vehicle Detection Project##

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Note: for those first two steps don't forget to normalize features and randomize a selection for training and testing.
* Implement a sliding-window technique and use a trained classifier to search for vehicles in images.
* Run a pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/samples_cars_not_cars.png
[image2]: ./output_images/hog_example.png
[image3]: ./output_images/96searchwindow.png
[image4]: ./output_images/sliding_window_testImages.jpg
[image5]: ./output_images/sliding_window_video.jpg
[image6]: ./output_images/sliding_window_heatmap.jpg
[image7]: ./output_images/static_results.png
[image8]: ./output_images/bboxes_and_heat.png
[image9]: ./output_images/labels_map.png
[image10]: ./output_images/output_bboxes.png
[video1]: ./project_result.mp4
[video2]: ./project_debug_result.mp4

---
###Histogram of Oriented Gradients (HOG)

####1. HOG feature extraction from the training images

The code for this step is in module `trainSvm.py` function `readDatabase()` at line 81 through 107. 
If `reducedSamples` parameter is `True`, only a randomized subset of 2000 samples is loaded and returned.

As database for the training I used the a combination of the GTI vehicle image database, the KITTI vision benchmark suite, and examples extracted from the project video itself offered on the project repository.

The database contains 8792 vehicles and 8968 non-vehicles.

I started by reading all the `vehicle` and `non-vehicle` images.  Here is an example of some images of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feeling for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(3, 3)`:

![alt text][image2]

####2. How to settle the final choice of HOG parameters

In module `trainSvm.py` I created a function `trainParamlist()` (line 191 through 214) that loops over a list of parameter combinations. It uses the above described randomized subset of the database and displays the accuracy on the trained model.

To execute the parameter search you call `python trainSvm.py task trainList1`.

First I examined for the combination of colorspaces (RGB, HSV, HLS, YUV, LUV, YCrCb), the combination of color space channels, the feature spaces (spatial,histogram,hog) 
I gained maximum classification accuracy when using RGB or HSL,  spatial and  hog featurs, but no histogram features.

The results for the different combinations are shown in the table below:

|Color space|Channels| Spatial | Histogram | HOG | Accuracy |
|:---------:|:------:|:-------:|:---------:|:---:|:--------:| 
| RGB | ALL | True | True | True | 0.9712 |
| RGB | ALL | False | True | True | 0.965 |
| RGB | ALL | True | False | True | 0.9638 |
| RGB | ALL | False | False | True | 0.9575 |
| RGB | ALL | True | True | False | 0.8625 |
| RGB | ALL | False | True | False | 0.4812 |
| RGB | ALL | True | False | False | 0.8938 |
| HSV | ALL | True | True | True | 0.9925 |
| HSV | ALL | False | True | True | 0.9912 |
| HSV | ALL | True | False | True | 0.99 |
| HSV | ALL | False | False | True | 0.9838 |
| HSV | ALL | True | True | False | 0.9775 |
| HSV | ALL | False | True | False | 0.9525 |
| HSV | ALL | True | False | False | 0.9062 |
| LUV | ALL | True | True | True | 0 |
| LUV | ALL | False | True | True | 0 |
| LUV | ALL | True | False | True | 0 |
| LUV | ALL | False | False | True | 0 |
| LUV | ALL | True | True | False | **0.9938** |
| LUV | ALL | False | True | False | 0.9788 |
| LUV | ALL | True | False | False | 0.98 |
| HLS | ALL | True | True | True | 0.9875 |
| HLS | ALL | False | True | True | 0.99 |
| HLS | ALL | True | False | True | 0.9925 |
| HLS | ALL | False | False | True | 0.9912 |
| HLS | ALL | True | True | False | 0.9838 |
| HLS | ALL | False | True | False | 0.9812 |
| HLS | ALL | True | False | False | 0.9075 |
| YUV | ALL | True | True | True | **0.995** |
| YUV | ALL | False | True | True | 0.9888 |
| YUV | 0 | True | False | True | 0.9812 |
| YUV | 1 | True | False | True | 0.985 |
| YUV | 2 | True | False | True | 0.9638 |
| YUV | ALL | False | False | True | 0.99 |
| YUV | ALL | True | True | False | 0.9638 |
| YUV | ALL | False | True | False | 0.4812 |
| YUV | ALL | True | False | False | 0.9512 |
| YCrCb | ALL | True | True | True | **0.9938** |
| YCrCb | ALL | False | True | True | 0.99 |
| YCrCb | ALL | True | False | True | **0.995** |
| YCrCb | ALL | False | False | True | 0.9875 |
| YCrCb | ALL | True | True | False | 0.965 |
| YCrCb | ALL | False | True | False | 0.4812 |
| YCrCb | ALL | True | False | False | 0.965 |

I found best result for RGB, HLS; HSV and YCrCb color space, histogram feature did not have a large impact.
 
In a second round I trained the best color spaces for combinations of their color channels using hog only. 
To execute this call `python trainSvm.py task trainList2`.

Only YCrCb showed a better result, when reducing the color channels.

|Color space|Channels| Spatial | Histogram | HOG | Accuracy |
|:---------:|:------:|:-------:|:---------:|:---:|:--------:| 
| YUV | ALL | False | False | True | **0.9862** |
| YUV | 0 | False | False | True | 0.9638 |
| YUV | 1 | False | False | True | 0.9238 |
| YUV | 2 | False | False | True | 0.885 |
| YUV | 0,1 | False | False | True | 0.9775 |
| YUV | 0,2 | False | False | True | **0.9862** |
| YUV | 1,2 | False | False | True | 0.94 |
| YCrCb | All | False | False | True | 0 |
| YCrCb | 0 | False | False | True | 0.965 |
| YCrCb | 1 | False | False | True | 0.9288 |
| YCrCb | 2 | False | False | True | 0.9112 |
| YCrCb | 0,1 | False | False | True | **0.99** |
| YCrCb | 0,2 | False | False | True | **0.9875** |
| YCrCb | 1,2 | False | False | True | 0.9475 |

LUV was not able to train because of NaN values in the scaler that caused exceptions. I did not look into that as I got high accuracies on other parmeters
YUV color channel 0 and 2 resulted in a high accuracy which offers a smaller feature size and faster processing.
The same applied for YCrCb color channel 0 and 1
In the last round I trained these other feature combinations.
To execute this call `python trainSvm.py task trainList3`.


|Color space|Channels| Spatial | Histogram | HOG | Accuracy |
|:---------:|:------:|:-------:|:---------:|:---:|:--------:| 
| YUV | ALL | True | True | True | **0.9925** |
| YUV | 0,2 | True | True | True | 0.9912 |
| YUV | 0,2 | False | True | True | 0.9838 |
| YUV | 0,2 | True | False | True | **0.9912** |
| YUV | 0,2 | False | False | True | 0.9875 |
| YUV | 0,2 | True | True | False | 0.95 |
| YUV | 0,2 | False | True | False | 0.4812 |
| YUV | 0,2 | True | False | False | 0.9625 |
| YUV | 0,2 | False | False | False | 0 |
| YCrCb | ALL | True | True | True | **0.9962** |
| YCrCb | 0,1 | True | True | True | 0.9875 |
| YCrCb | 0,1 | False | True | True | 0.99 |
| YCrCb | 0,1 | True | False | True | **0.9938** |
| YCrCb | 0,1 | False | False | True | 0.9862 |
| YCrCb | 0,1 | True | True | False | 0.9625 |
| YCrCb | 0,1 | False | True | False | 0.4812 |
| YCrCb | 0,1 | True | False | False | 0.9588 |
| YCrCb | 0,1 | False | False | False | 0 |

Interestingly the results vary always a bit even when defining a random seed. I chosed YCrCb and 0,1 color channel spatial and hog features. Its has a little smaller feature size, but and accuracy of 99.38%. The feature vector has a length of 6600. The accuracy for the choosen parameters is 0.9867 for all images.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the class SGDClassifier. In the main function at `trainSvm.py` line 322 through 370. The already loaded car and not-car full datasets and the train parameters are handed over to the function `train()` at line 260 through 277. To execute this call `python trainSvm.py task best`.

First the features are extracted in `getFeatures()` at line 214 through 257 calling the `extractFeatures()` function in module `featureDetection.py`. The two feature sets are stacked in a two line matrix calling vstack. A label column is added to the feature matrix(label 1 for cars and label 0 for notcars) in lines 249.

In line 245 through 247 the feature matrix is normalized to 0 mean and unit variance using the class sclearn.prepocessing.StandardScaler. The `fit()` method calculates the mean and standard deviation, which is used for scaling. Calling `transform()`, the data is scaled. The scaler has to be saved together with a trained model to allow a reuse. data for predictiosn has to be scaled identically to the training data.

The feature matrix is devided in a randomized 80% train and 20% test dataset calling `train_test_split()` at line 252.

After the preprocessing the Classifier is created in line 265 and trained in line 266.

The function `train()` returns the scaler object, the trained classifier, the achieved accuracy and the feature parameters. As the scaler and the trained classifier are needed for future predictions, a dictionary is created and stored to a pickle file named `svc.p` at line 358 to 363.

###Sliding Window Search

####1. Implementation of a sliding window search.  Choice of scales and overlap of windows.
The search windows are created in function `slideWindows()` in module `featureDetection.py` at line 140 through 189.

The function creates for an area of interest, for a defined scale and an overlap ratio as many windows as fit into the search area. I expected to use larger scales for closer object (== higher y values) and smaller scales for more distant objects (== lower y values).

I choosed always quadratic windows, as the database images are as well quadratic and the perspective tranformation should be linear on both axis.

I played with different scales and found that three scales (64,96 and 128) are sufficient for reliable detections.

The overlap should not to be too big in an large search area, as it increases the amount of search windows and therefore computation costs.

I choosed as area of interest for y values between 390 and 660 to cover the lane area and to surpress the visible part of the car cockpit.

For the test images, I used an area of interest from left to right as there is no history of detections available.
Overlay is between 0.66 and 0.5. Using the overlap the vehicles usually produce more than one hit and therefore can be better separated from false detections.

![alt text][image4]

For the video processing, I choosed an area at the low left and low right as entry area for overtaking cars. I added a high center area for vehicles that are over taken.

![alt text][image5]

The below described heat map is used to provide small search areas, where cars have been detected in previous images and the future position is forecasted.
I defined a symetric grid around the expected car position to track it through the images stream. The grid is created at `getSearchAreas()` in the module `heatmap.py` line 98 through 126 and uses scales 64, 96 and 128 pixel having an overlap of 0.9. Again I choose a high overlap to reduce the false error rate.

![alt text][image6]

####2. Demonstration of the process pipeline on test images. Optimization of the performance of the classifier?
Initially in the main function of module the trained model, the scaler and the feature parameter are loaded at line 128 through 132 in module `processChain.py`. 

The pipeline for the processing is defined in the method `processColoredImage()` module `processChain.py` at line 15 through 94.

First the image is resized, if required at line 21. Then the image is scaled between 0 and 1 following the training images format.

In line 33 through 42 the search parameters for the sliding windows are set and the window list is created calling `slideWindow()` of module `featureDetection.py`.

The search area retrieved from the heat map is added to the window search list at line 45 calling `getSearchAreas()` of module `heatmap.py`. This function takes all forecasted car positions and lays a grid around these positions.

The window list is searched at line 59 calling `searchWindows()` of module `featureDetection.py` using the same parameter as for the training of the classifier.

The function `searchWindows()` loops over the windows list, resizes the image to (64,64) scale like the database images. It then creates the feature vector and normalizes it using the same scales as used in the training part.

In the function I added a  flag `hardNegative`. If this is activated all windows predicted as label `car' are stored in a separate folder. This images are then manually verified to add them to the non-vehicle or vehicle database folder. Executing a retraining improves significantly the accuracy of the classifier in 1 to 3 cycles.

The results of the windows search is used to update a heat map at line 73 in module `processChain.py` 
calling `update()` of the class `Heatmap` as defined in module `heatmap.py`.

For each pixel in any window the heat value is increased by one. The resulting boxes for detected cars is calulated calling `calcBoxes()` at line 78. 

In this function I call `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap assuming each blob correspondeds to a vehicle. I constructed bounding boxes to cover the area of each blob detected. 

I choosed the SGDClassifier as it is fast and has more options to optimize. For this reason I created a function `gridSearch` in the module `trainSvm.py`. This estimator implements regularized linear models with stochastic gradient descent (SGD) learning.

It allows to find the best setup for the classifier looping a multi-dimensional parameter space.
I defined the parameter space for the tuning by `[{'loss':["hinge","modified_huber","squared_hinge"],'alpha': [0.00001,0.0001,0.001,0.01],"penalty":["l1","l2","elasticnet"]}]`

The found optimal parameters are: loss=`squared_hinge`,`penalty=`elasticnet` and `alpha=`0.01`. The alpha parameter is 1/C for usual Linear SVM.

I decided have large overlay of search windows around the expected car position to improve the signal / error ratio. The more hit are stored in the heat map, the less false detections occure. 
For sure this costs computation power, but as I kept the search area  small, the performance is still good.

![alt text][image7]
---

### Video Implementation

####1. Link to the final video output. 
Here is my result of the vehicle detection:

![alt text][video1]

####2. Filter for false positives and combining overlapping bounding boxes.
The processing is nearly identical to the pipeline described fopr the test images. Main difference is the use of an averaged heat map. The heat values are stored in a list for the 10 recent images. An averaged heat map for these history of images is calculated by the mean calling `heatmap.average()` at line 77. 

Additionally the movement of a box representing a vehicle is retrieved by subtracting from the last detected position at line 69 through 74 in module `heatmap.py`. I use this to fore cast the car position in the next image.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

The heat map and the dynamic grid around the detected cars is shown in a debug video

![alt text][video2]

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

