import os
import pickle
import cv2
import numpy as np
import scipy.misc

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm

#find 3D object points in an image and transform them to 2D image points. 
def getChessPoints(gray, chessGrid, objpoints,imgpoints):
  (nx, ny) = chessGrid 
  # find chessboard points
  ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None )

  # if found add image point and object points to lists
  if ret and not corners is None:
    objp = np.zeros((nx*ny,3),np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)  

    objpoints.append(objp)
    imgpoints.append(corners)
    return True
  return False

#calibrate a camera using images in an directory
def calibrateCamera(images_dir,image_shape,chessgrids):

  objpoints = [] # 3d point in real world space
  imgpoints = [] # 2d points in image plane.

  count_ok = 0
  # get list of files in an directory
  images = os.listdir(images_dir)
  for i, item in enumerate(tqdm(images, desc='Processing image')):    
    path_to_file = os.path.join(images_dir,item)
    #read the image
    image = mpimg.imread(path_to_file) 
    
    #resize it, if required
    if image.shape[0] != image_shape[0] or image.shape[1] != image_shape[1]:
      image = cv2.resize(image,(image_shape[1],image_shape[0]))    

    # Convert to grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    # blur the image
    gray = cv2.medianBlur(gray,5)

    #not all images can be calibrated using a (6 / 9), try a list of possible grids
    for chessgrid in chessgrids:
      #find points in 2D and 3D space
      ret = getChessPoints(gray,chessgrid,objpoints,imgpoints)
    
      if ret:
        count_ok += 1
      break
    
  if count_ok > 0:
    #use the points find a transformation for camera calibration    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape ,None,None)
    
    calibration = {'mtx': mtx,
               'dist': dist,
               'shape':IMAGE_SIZE}

    #save the calibration in a pickle file
    with open(CALIBRATION_PATH, 'wb') as f:
      pickle.dump(calibration, file=f)    
        
    return (mtx,dist)
  else:
    return None,None

#show the original and the undistored image using a calibration
def showCalibrateImage(image,mtx,dist):

  f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
  f.tight_layout()
  ax1.imshow(image)
  ax1.set_title('Original Image', fontsize=50)

  undist = cv2.undistort(image, mtx, dist, None, mtx)    
  ax2.imshow(undist)
    
  ax2.set_title('Calibrated Image', fontsize=50)
  plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
  #loop over the test pictures
  plt.show()

def getCalibration():
  with open(CALIBRATION_PATH, "rb") as f:
    calibration = pickle.load(f)
  mtx = calibration["mtx"]
  dist = calibration["dist"]
  image_shape = calibration["shape"]
  return (mtx,dist,image_shape)

CALIBRATION_PATH = "./calibration.p"

if __name__ == '__main__':

  images_dir = "camera_cal"
  output_dir = "output_images"
  IMAGE_SIZE = (720,1280)
  CHESS_GRIDS = ((6,9),(5,9))

  mtx,dist = calibrateCamera(images_dir,IMAGE_SIZE,CHESS_GRIDS)

  print("mtx:",mtx)
  print("dist:",dist)

  #generate the undistorded images
  images = os.listdir(images_dir)
  for item in images:    
    path_to_file = os.path.join(images_dir,item)

    image = mpimg.imread(path_to_file) 
#    showCalibrateImage(image,mtx,dist)#to show calibrated images

    #undistord using the calibration
    image = cv2.undistort(image, mtx, dist,None,mtx)    

    #save the result
    path_to_image = os.path.join(output_dir,"{0}_{1}.jpg".format(item[:-4],"undist"))
    scipy.misc.imsave(path_to_image, image)


