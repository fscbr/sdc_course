import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import GridSearchCV
from skimage.feature import hog
import scipy.misc
import pickle
import featureDetection as fd

# NOTE: the next import is only valid for scikit-learn version <= 0.17
from sklearn.cross_validation import train_test_split
# for scikit-learn >= 0.18 use:
#from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import numpy as np
    
#answer a sample or all database images
def displayDatabaseSample():    
  # Read in cars and notcars
  images = glob.glob('database/vehicles/*/*.png')
  cars = []
  notcars = []
  for image in images:
    cars.append(image)

  images = glob.glob('database/non-vehicles/*/*.png')
  for image in images:
    notcars.append(image)

  np.random.seed(1)
  sample_size = 32
  random_indizes = np.arange(sample_size)
  np.random.shuffle(random_indizes)

  cars = np.array(cars)[random_indizes]
  notcars = np.array(notcars)[random_indizes]

  #show one image for 20 samples
  row = 0
  col = 0
  fig = plt.figure()
  size = 6
  gs = gridspec.GridSpec(size, size)
#  fig.subplots_adjust(top=1.5)
  for j in range(2):
   if j == 0:
     data = cars
     title = "car"
   else: 
     data = notcars
     title = "no car"
    
   for i in range(int(size*size/2)):
    image = mpimg.imread(data[i])
    a = plt.subplot(gs[row, col])
    a.imshow(image.copy())
    a.get_xaxis().set_visible(False)
    a.get_yaxis().set_visible(False)
    a.set_title(title)
    fig.add_subplot(a)
    col += 1                   
    if col == size:
      col = 0
      row += 1
            
  fig = plt.gcf()
  fig.savefig("output_images/samples_cars_not_cars.png") 
  plt.show()

#answer a sample or all database images
def readDatabase(reducedSamples):    
  # Read in cars and notcars
  images = glob.glob('database/vehicles/*/*.png')
  cars = []
  notcars = []
  for image in images:
    cars.append(image)

  images = glob.glob('database/non-vehicles/*/*.png')
  for image in images:
    notcars.append(image)

  image = mpimg.imread(cars[0])
  print("min:",np.min(image[0])," max:",np.max(image[0]))
  
  if reducedSamples:    
    np.random.seed(1)
  # Reduce the sample size for fast testing
    sample_size = 2000
    random_indizes = np.arange(sample_size)
    np.random.shuffle(random_indizes)

    cars = np.array(cars)[random_indizes]
    notcars = np.array(notcars)[random_indizes]

  print("cars:",len(cars)," not cars:",len(notcars))
  return (cars,notcars)

#answer a list of param for optimization
def getParamlist3():
  params = (
    ("YCrCb","ALL",True,True,True,3),
    ("YCrCb","0,1",True,True,True,3),
    ("YCrCb","0,1",False,True,True,3),
    ("YCrCb","0,1",True,False,True,3),
    ("YCrCb","0,1",False,False,True,3),
    ("YCrCb","0,1",True,True,False,3),
    ("YCrCb","0,1",False,True,False,3),
    ("YCrCb","0,1",True,False,False,3),
    ("YCrCb","0,1",False,False,False,3))
  return params

def getParamlist2():
  params = (
    ("YUV","ALL",False,False,True,3),
    ("YUV",0,False,False,True,3),
    ("YUV",1,False,False,True,3),
    ("YUV",2,False,False,True,3),
    ("YUV","0,1",False,False,True,3),
    ("YUV","0,2",False,False,True,3),
    ("YUV","1,2",False,False,True,3),
    ("YCrCb","All",False,False,True,3),
    ("YCrCb",0,False,False,True,3),
    ("YCrCb",1,False,False,True,3),
    ("YCrCb",2,False,False,True,3),
    ("YCrCb","0,1",False,False,True,3),
    ("YCrCb","0,2",False,False,True,3),
    ("YCrCb","1,2",False,False,True,3))
  return params

def getParamlist():
  params = (
    ("RGB","ALL",True,True,True),
    ("RGB","ALL",False,True,True),
    ("RGB","ALL",True,False,True),
    ("RGB","ALL",False,False,True),
    ("RGB","ALL",True,True,False),
    ("RGB","ALL",False,True,False),
    ("RGB","ALL",True,False,False),
    ("HSV","ALL",True,True,True),
    ("HSV","ALL",False,True,True),
    ("HSV","ALL",True,False,True),
    ("HSV","ALL",False,False,True),
    ("HSV","ALL",True,True,False),
    ("HSV","ALL",False,True,False),
    ("HSV","ALL",True,False,False),
    ("LUV","ALL",True,True,True),
    ("LUV","ALL",False,True,True),
    ("LUV","ALL",True,False,True),
    ("LUV","ALL",False,False,True),
    ("LUV","ALL",True,True,False),
    ("LUV","ALL",False,True,False),
    ("LUV","ALL",True,False,False),
    ("HLS","ALL",True,True,True),
    ("HLS","ALL",False,True,True),
    ("HLS","ALL",True,False,True),
    ("HLS","ALL",False,False,True),
    ("HLS","ALL",True,True,False),
    ("HLS","ALL",False,True,False),
    ("HLS","ALL",True,False,False),
    ("YUV","ALL",True,True,True),
    ("YUV","ALL",False,True,True),
    ("YUV","ALL",True,False,True),
    ("YUV","ALL",False,False,True),
    ("YUV","ALL",True,True,False),
    ("YUV","ALL",False,True,False),
    ("YUV","ALL",True,False,False),
    ("YCrCb","ALL",True,True,True),
    ("YCrCb","ALL",False,True,True),
    ("YCrCb","ALL",True,False,True),
    ("YCrCb","ALL",False,False,True),
    ("YCrCb","ALL",True,True,False),
    ("YCrCb","ALL",False,True,False),
    ("YCrCb","ALL",True,False,False))
  return params

#train a list of parameters
def trainParamlist(cars,notcars):
  params = getParamlist()

  results = {}
  #train the list
  for i in range(len(params)):
    try:
      (scaler,svc,accuracy) = train(params[i],cars,notcars)
      results[params[i]] = accuracy
    except ValueError: 
      results[params[i]] = 0

  #print result as wiki table
  print("|Color space| Channels | Spatial | Histogram | HOG | Accuracy |")
  best = sorted(results, key=results.get, reverse=True)[0:min(4,len(results))]
  for i in range(len(params)):
    key = params[i]
    accuracy = results[key]
    
    (color_space,hog_channel,spatial_feat,hist_feat,hog_feat) = key

    if key in best:
      print("|",color_space,"|",hog_channel,"|",spatial_feat,"|",hist_feat,"|",hog_feat,"| **",accuracy,"** |")
    else:
      print("|",color_space,"|",hog_channel,"|",spatial_feat,"|",hist_feat,"|",hog_feat,"|",accuracy,"|")
    
    
#train once
def train(param,cars,notcars):
    
  orient = 9  # HOG orientations
  pix_per_cell = 8 # HOG pixels per cell
  spatial_size = (16, 16) # Spatial binning dimensions
  hist_bins = 16    # Number of histogram bins

  (color_space,hog_channel,spatial_feat,hist_feat,hog_feat,cell_per_block) = param
#  print(i," params:", params[i])  

  car_features = fd.extractFeatures(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat,feature_vec=True)

  notcar_features = fd.extractFeatures(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat,feature_vec=True)

  X = np.vstack((car_features, notcar_features)).astype(np.float64)      
  print(X.shape)
# Fit a per-column scaler
  X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
  scaled_X = X_scaler.transform(X)
# Define the labels vector
  y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
#Split up data into randomized training and test sets
  rand_state = 1
  X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)
  print(param)
  print(' using:',orient,'orientations',pix_per_cell,'pixels per cell and', cell_per_block,'cells per block -> feature vector length:', len(X_train[0]))

#train
  svc = SGDClassifier(fit_intercept=False, loss="hinge", n_jobs=-1, learning_rate="optimal", penalty="elasticnet", class_weight="balanced",n_iter=10)
  svc.fit(X_train, y_train)

#get the accuracy      
  accuracy = round(svc.score(X_test, y_test), 4)
  print('Test Accuracy of SVC = ', accuracy)

  return (X_scaler,svc,accuracy)

SVC_PATH = "./svc.p"
#load a pickle file and return the model dictionary  containg the keys X_scaler and svc
def getModelData():
  data = {}
  #load trained svc
  with open(SVC_PATH, "rb") as f:
    data = pickle.load(f)
  return data

  
if __name__ == '__main__':
  if False:
    displayDatabaseSample()
  #read a sample
  if False:
    print("read sample database")
    (cars,notcars) = readDatabase(True)
 
  #train all for optimization
    print("train all params")
    trainParamlist(cars,notcars)

  if True:  
#read all
    print("read full size database")
    (cars,notcars) = readDatabase(False)

#train the best choice
    param = ("YCrCb","0,1",True,False,True,3)
    print("train best param")
    (X_scaler,svc,acccuracy) = train(param,cars,notcars)

#save the calibration in a pickle file
    data = {}
    data["X_scaler"] = X_scaler
    data["svc"] = svc
    data["param"] = param
    with open(SVC_PATH, 'wb') as f:
      pickle.dump(data, file=f)    
    
  if False:
    scores = ['precision', 'recall']

# Set the parameters by cross-validation
    tuned_parameters = [{'loss':["hinge","modified_huber","squared_hinge"],'alpha': [0.00001,0.0001,0.001,0.01],"penalty":["l1","l2","elasticnet"]}]

    scores = ['precision', 'recall']

    for score in scores:
      print("# Tuning hyper-parameters for %s" % score)
      print()

#      clf = GridSearchCV(SGDClassifier(shuffle=True, fit_intercept=False, n_jobs=-1, learning_rate="optimal", penalty="l2", class_weight="balanced",n_iter=5), tuned_parameters, cv=5,
#                       scoring='%s_macro' % score)
#      clf.fit(X_train, y_train)

      print("Best parameters set found on development set:")
      print()
      print(clf.best_params_)
      print()
      print("Grid scores on development set:")
      print()
      means = clf.cv_results_['mean_test_score']
      stds = clf.cv_results_['std_test_score']
      for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
      print()

      print("Detailed classification report:")
      print()
      print("The model is trained on the full development set.")
      print("The scores are computed on the full evaluation set.")
      print()
      y_true, y_pred = y_test, clf.predict(X_test)
      print(classification_report(y_true, y_pred))
      print()

      