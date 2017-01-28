import os
import pickle
import cv2
import numpy as np
import scipy.misc

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.figure import Figure
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


#extract the lower center part of the image
def filterAreaOfInterest(image):
    maxx = image.shape[1]
    maxy = image.shape[0]
    y1 = int(maxy)
    y2 = int(maxy*0.58)
    x1 = int(maxx*0.4)
    x2 = int(maxx*0.6)
    x3 = int(maxx)
    area = np.array([[0,y1],[x1,y2],[x2,y2],[x3,y1]])
    
    return region_of_interest(image,[area])

#binary from RGB colorspace image using a dynamic threshold calulated from the 99% quantileon for each color channel
def binaryFromRGB(image):
  threshold = np.percentile(image[:,:,:],99)

# Use a "bitwise AND" to identify pixels above the threshold
  binary = np.zeros_like(image[:,:,0])
  binary[(image[:,:,0] >= threshold) \
            & (image[:,:,1] >= threshold) \
            & (image[:,:,2] >= threshold)]= 255

  return binary

#binary image combining gray and YUV channels for yellow lines and RGB color space for white lines. Combined using "OR" operator
def binaryFromColorSpaces(image):
  image = cv2.blur(image,(15,15))
  c_binary = binaryFromRGB(image)
  g_binary = binaryFromYUV(image)
  binary = np.zeros_like(g_binary)
  binary[(c_binary > 0) | (g_binary > 0)] = 255    
    
  return binary

#create a one channel image build from the inverse V chrominance      
def imageFromYUV(image):
  yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
  gray = np.zeros_like(image[:,:,0])
  gray = 255 - yuv[:,:,2]

  return gray 

#binary from YUV colorspace image using a dynamic threshold calulated from the 99% quantile on the color channel
def binaryFromYUV(image):
  V = imageFromYUV(image)

  threshold = np.percentile(V[:,:],99)

  binary = np.zeros_like(V[:,:])
  binary[(V[:,:] >= threshold)]= 255
  return binary


