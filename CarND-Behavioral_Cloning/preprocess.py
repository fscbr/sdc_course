from PIL import Image
from PIL import ImageOps
import cv2
import numpy as np

#mask an image using a polygon. In this variant the polygon is used to wipe out
def region_of_interest(img, vertices):
    #defining a white mask to start with
    mask = np.zeros_like(img) +255  
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (0,) * channel_count
    else:
        ignore_mask_color = 0
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

#wipe out the lower and center part of the image as we have clutter there
def filterAreaOfInterest(image):
    maxx = image.shape[1]
    maxy = image.shape[0]

    x1 = int(maxx*0.1)
    y1 = int(maxy)
    y2 = int(maxy*0.33)
    x2 = int(maxx*0.33)
    x3 = int(maxx*0.66)
    x4 = int(maxx*0.9)
    area = np.array([[x1,y1],[x2,y2],[x3,y2],[x4,y1]])
    
    return region_of_interest(image,[area])

#extract image content by finding the contour of features in the image
def extractContour(image):

  img = image.copy()

  # Otsu's thresholding after Gaussian filtering
  blur = cv2.GaussianBlur(img,(7,7),0)
  ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)    

  #find contours
  im2, contours, hierarchy = cv2.findContours(th3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  contImg = np.zeros(image.shape)

  #filter contours having a minimum lenght of 100 to reduce clutter
  contoursFiltered = []
  for i in range(len(contours)):
    if contours[i].shape[0] > 100:
      contoursFiltered.append(contours[i])

  #draw contours on a blank image
  cv2.drawContours(contImg, contoursFiltered, -1, (255,255,255), 2)

  #wipe out center low area to reduce clutter
  contImg = filterAreaOfInterest(contImg)
  return contImg  

#randomly shift vertical and horizontal an image in a range 
#range of y is half of range of x
#no change of steering
def trans_image(image,trans_range):
    # Translation
    tr_x = int(trans_range*np.random.uniform()-trans_range/2)
    tr_y = int(trans_range*np.random.uniform()-trans_range/4)

    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(image.shape[1],image.shape[0]))
    
    return image_tr

def preprocess_image(image):
#reduce the image on the interesting part
  image = image[60:140,:]

#get gray and hls color spaces  
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

#use a three layer image having gray, red and saturation. Extract contours on them
  image[:,:,0] = extractContour(gray)#gray
  image[:,:,1] = extractContour(image[:,:,0])#red
  image[:,:,2] = extractContour(hls[:,:,2])#saturation

#resize the image for the model
  image = cv2.resize(image, (64,64))
  image = image.astype("float32")

#normalize the values between -1 and 1
  image /= 127.5
  image -= 1
  return image

