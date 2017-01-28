import os
import pickle
import cv2
import numpy as np
import scipy.misc

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import calibration as cb

#applies a perspective transformation to an image
def transformPerspective(image,M):
  warped = cv2.warpPerspective(image, M, (image.shape[1],image.shape[0]), flags=cv2.INTER_LINEAR)
  return warped

#Calculates a perspective transform from four pairs of the corresponding points in both directions
def calcCameraPerspective(image,src,dst):
  M = cv2.getPerspectiveTransform(src, dst)
  Minv = cv2.getPerspectiveTransform(dst, src)  

  return M,Minv

#displays the orginal and the transform image 
def showPerspectiveTransformation(image,warped,SRC,DST,path_to_image):
  f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
  f.tight_layout()

  cv2.line(image, (SRC[0][0],SRC[0][1]), (SRC[1][0],SRC[1][1]), [255,0,0], 5)
  cv2.line(image, (SRC[2][0],SRC[2][1]), (SRC[3][0],SRC[3][1]), [255,0,0], 5)

  cv2.line(warped, (DST[0][0],DST[0][1]), (DST[1][0],DST[1][1]), [0,255,0], 5)
  cv2.line(warped, (DST[2][0],DST[2][1]), (DST[3][0],DST[3][1]), [0,255,0], 5)

  ax1.imshow(image)
  ax1.set_title('Test Image to gain Perspective', fontsize=50)

  ax2.set_title('Transformed Image', fontsize=50)
  ax2.imshow(warped)

  plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
  plt.show()    
  f.savefig(path_to_image)

def getPerspective():
  with open(PERSPECTIVE_PATH, "rb") as f:
    perspective = pickle.load(f)

  M = perspective["M"]
  Minv = perspective["Minv"]
  image_shape = perspective["shape"]
  return (M,Minv,image_shape)



OFFSET = 150
SRC = np.float32([[ 235,692],[ 584,448],[ 702,448],[ 1080,692]])
DST = np.float32([[ 235+OFFSET,720],[ 235+OFFSET,0],[ 1080-OFFSET,0],[ 1080-OFFSET,720]])

INPUT_DIR = "test_images"
OUTPUT_DIR = "output_images"
IMAGE_NAME = "solidWhiteRight"
CALIBRATION_PATH = "./calibration.p"
PERSPECTIVE_PATH = "./perspective.p"

if __name__ == '__main__':
  #load calibration
  (mtx,dist,image_shape) = cb.getCalibration()
 
  #load test image
  path_to_image = os.path.join(INPUT_DIR,"{0}.jpg".format(IMAGE_NAME))
  image = mpimg.imread(path_to_image)    

  #resize test image
  image = cv2.resize(image,(image_shape[1],image_shape[0]), interpolation = cv2.INTER_CUBIC)
  path_to_image = os.path.join(OUTPUT_DIR,"{0}_{1}_{2}.jpg".format(IMAGE_NAME,image_shape[1],image_shape[0]))
  scipy.misc.imsave(path_to_image, image)

  #undistort
  image = cv2.undistort(image, mtx, dist, None, mtx)    
  path_to_image = os.path.join(OUTPUT_DIR,"{0}_{1}_{2}_undist.jpg".format(IMAGE_NAME,image_shape[1],image_shape[0]))
  scipy.misc.imsave(path_to_image, image)

  #calculate cameraPerpective
  M,Minv = calcCameraPerspective(image,SRC,DST)
  print(M)

  perspective = {'M': M,
               'Minv': Minv,
               'shape':image_shape}

  #save the perspective in a pickle file
  with open(PERSPECTIVE_PATH, 'wb') as f:
    pickle.dump(perspective, file=f)    

  #applies the perspective transformation to an image  
  warped = transformPerspective(image,M)
  path_to_image = os.path.join(OUTPUT_DIR,"{0}_{1}_{2}_warped.jpg".format(IMAGE_NAME,image_shape[1],image_shape[0]))
  scipy.misc.imsave(path_to_image, warped)

  #display the orginal and the transformed image
  path_to_image = os.path.join(OUTPUT_DIR,"{0}_{1}_{2}_check.jpg".format(IMAGE_NAME,image_shape[1],image_shape[0]))
  debug = showPerspectiveTransformation(image,warped,SRC,DST,path_to_image)

