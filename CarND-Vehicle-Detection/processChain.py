import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import scipy.misc
import pickle
import heatmap as hm
import trainSvm as ts
import featureDetection as fd
import globalData as gd

#the process chain for an image. 
def processColoredImage(image,smooth,debug,imageName,X_scaler,svc,param):

  heatmap = gd.data["heatmap"]
  gd.data["image_counter"]= gd.data["image_counter"] + 1
  
#resize if required  
  if image.shape[0]!= 720 or image.shape[1]!= 1280:
    image = cv2.resize(image,(1280, 720), interpolation = cv2.INTER_CUBIC)
    print("resize:",image.shape)

#png images need rescaling 
  normImage = image.copy().astype("float32") / 255.0

  windows = []
  result = image
  debugImg = image

#the search area will be created by using the history of detecuions (video mode)
  search_param = getSearchParam(smooth)
  all_hot_windows = []
#  create the windows list sliding ofver the search area
  for i in range(len(search_param)):  
    (size,y_start, y_stop, x_start, x_stop,ov) = search_param[i]

    new_win = fd.slideWindow(normImage, x_start_stop=(x_start, x_stop), y_start_stop=(y_start, y_stop), 
                    xy_window=(size,size), xy_overlap=(ov, ov))
    
    windows += new_win

  if smooth:
    windows += heatmap.getSearchAreas(normImage)
    if debug:
      result = fd.drawBoxes(result, windows, color=(255,255,255), thick=(1))  
  else:
#static mode for test images  
    heatmap = hm.Heatmap(1)
   
#  print(windows2)
  (color_space,hog_channel,spatial_feat,hist_feat,hog_feat,cell_per_block) = param
  orient = 9  # HOG orientations
  pix_per_cell = 8 # HOG pixels per cell
  spatial_size = (16, 16) # Spatial binning dimensions
  hist_bins = 16    # Number of histogram bins

# classify the image parts in the search windows
  hot_windows = fd.searchWindows(normImage, windows, svc, X_scaler, color_space=color_space, 
                      spatial_size=spatial_size, hist_bins=hist_bins, 
                      orient=orient, pix_per_cell=pix_per_cell, 
                      cell_per_block=cell_per_block, 
                      hog_channel=hog_channel, spatial_feat=spatial_feat, 
                      hist_feat=hist_feat, hog_feat=hog_feat,hardNegative=False)                       

  if debug:
#draw the search window grid for debugging                    
    result = fd.drawBoxes(result, windows, color=(0,255,255), thick=(2))  
#draw the windows that got a match for debugging                      
    debugImg = fd.drawBoxes(debugImg, hot_windows, color=(0, 0, 255), thick=1)                    

#update the heatmap    
  heatmap.update((image.shape[0],image.shape[1]),hot_windows)
    
  if smooth:
#average the heatmap over the history of updates  
    heatmap.average()
    heatmap.calcBoxes(result)
    result = heatmap.drawLabeledBoxes(result,(0,255,0),True)
    if debug:
      debugImg = heatmap.drawLabeledBoxes(debugImg,(0,255,0),True)
  else:
    heatmap.calcBoxes(result)
    result = heatmap.drawLabeledBoxes(result,(255,0,0),False)
    if debug:
      debugImg = heatmap.drawLabeledBoxes(debugImg,(255,0,0),True)
    
  if debug:
    if not imageName is None:
      path_to_image = os.path.join("output_images","{0}_{1}.jpg".format(imageName,"debug"))
      scipy.misc.imsave(path_to_image, debugImg)    
    
  return result

#define search parameter for window search.
def getSearchParam(isSmooth):
  if isSmooth:
    search_param = ((64,390,498,224,1050,0.66),
     (96,390,534,38,278,0.5),
     (96,390,534,1002,1242,0.5),
     (160,444,659,10,279,0.66),
     (160,444,659,1002,1271,0.66),
     (192,430,622,10,333,0.66),
     (192,430,622,948,1271,0.66))  
  else:
    search_param = ((64,390,650,20,1260,0.66),
      (96,390,552,21,1260,0.66),
      (128,390,606,10,1270,0.66))  
  return search_param



      
if __name__ == '__main__':

  #init global data
  gd.data["image_counter"]=0
  gd.data["heatmap"]=hm.Heatmap(1)
   
  print(os.listdir("test_images/"))
  #loop over the test pictures
  resultPath = "result_images/"
  if not os.path.exists(resultPath):
    os.mkdir(resultPath)

#load the model     
  data = ts.getModelData()
  svc = data["svc"]
  X_scaler = data["X_scaler"]
  svc = data["svc"]
  param = data["param"]

  #loop over the test pictures
  item = "test3.jpg"
#  if True:
  for item in os.listdir("test_images/"):
    print(item)
    path_to_image = os.path.join("test_images",item)
    image = mpimg.imread(path_to_image)    

#call the process chain 
    result = processColoredImage(image,True,True,item[:-4],X_scaler, svc, param)

#display origin and result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=50)

    ax2.set_title('Processed Image', fontsize=50)
    ax2.imshow(result)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

#save origin and result
    path_to_result = os.path.join(resultPath,"result_{0}".format(item))
    fig = plt.gcf()
    fig.savefig(path_to_result) 
    plt.show()
