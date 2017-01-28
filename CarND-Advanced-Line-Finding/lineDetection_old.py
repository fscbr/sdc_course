import os
import pickle
import cv2
import numpy as np
import scipy.misc

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.figure import Figure
import math
import calibration as cb
import cameraPerspective as cp
import line
import imageProcessing as ip

def drawLinesOnRoad(image,Minv,warped,lfx,ly,rfx,ry):

# Create an image to draw the lines on
  warp_zero = np.zeros_like(warped).astype(np.uint8)
  color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Recast the x and y points into usable format for cv2.fillPoly()
  pts_left = np.array([np.transpose(np.vstack([lfx, ly]))])
  pts_right = np.array([np.flipud(np.transpose(np.vstack([rfx, ry])))])
  pts = np.hstack((pts_left, pts_right))
# Draw the lane onto the warped blank image
  cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

# Warp the blank back to original image space using inverse perspective matrix (Minv)
  newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
# Combine the result with the original image
  result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
  return result  

def showBinaryAndWarped(binary,warped,lx,ly,rx,ry,lfx,lfy,rfx,rfy):
  f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
  f.tight_layout()
  ax1.imshow(binary,cmap="gray")
  ax1.set_title('color filter', fontsize=50)

  ax2.set_title('warped image', fontsize=50)
  ax2.imshow(warped,cmap="gray")
  ax2.plot(lx, ly, 'o', color='red')
  ax2.plot(rx, ry, 'o', color='blue')

  ax2.plot(lfx, lfy, 'o-', color='yellow')
  ax2.plot(rfx, rfy, 'o-', color='green')
  plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
  plt.show()

def showDebug(warped,lx,ly,rx,ry,lfx,lfy,rfx,rfy):
  global image_counter

  painted = warped.copy()
  painted = cv2.cvtColor(painted, cv2.COLOR_GRAY2RGB)

  #draw image counter
  font = cv2.FONT_HERSHEY_SIMPLEX
  cv2.putText(painted, "Image: {0}".format(image_counter), (50, 50), font, 1, (255, 255, 255), 2)

  pts = zip(lfx,lfy)
  for pt in pts:    
    cv2.circle(painted, pt, 3, (255,0,0),2)
  pts = zip(rfx,rfy)
  for pt in pts:    
    cv2.circle(painted, pt, 3, (0,255,0),2)
  pts = zip(lx,ly)
  for pt in pts:    
    cv2.circle(painted, pt, 3, (0,0,255),2)
  pts = zip(rx,ry)
  for pt in pts:    
    cv2.circle(painted, pt, 3, (0,255,255),2)

  return painted

def createCenterLine(image,leftLine,rightLine):
  lfxs,lfys = leftLine.getSmoothed()
  rfxs,rfys = rightLine.getSmoothed()
    
  cx = (lfxs+rfxs)/2
  centerLine = line.Line()
  centerLine.create(cx,lfys)
  centerLine.calcCurveRad()    
  centerLine.calcOffset(image)
  return centerLine

#the process chain to draw lane lines on the road
def process_colored_image(image,image_shape,mtx,dist,M,Minv,smooth,debug,imageName):
  global image_counter,leftLine,rightLine
  image_counter += 1
  
#  if image_counter == 627:
#    saveImageAndReturn(image, "testImage", image_counter)    

  #avoid history building
  if not smooth:
    leftLine = line.Line() 
    rightLine = line.Line() 

  #resize if required
  if image.shape[0]!= image_shape[0] or image.shape[1]!= image_shape[1]:
    image = cv2.resize(image,(image_shape[1], image_shape[0]), interpolation = cv2.INTER_CUBIC)
    print("resize:",image.shape)

  #undistort
  undist = cv2.undistort(image, mtx, dist, None, mtx)    

  # apply an area of interest
  area_filtered_image = ip.filterAreaOfInterest(undist)
  warped = cp.transformPerspective(area_filtered_image,M)

  binary = ip.binaryFromColorSpaces(warped)

  # move to gray shaded
  gray = ip.imageFromYUVGray(warped)
  # apply the mask to the gray shaed image
  masked_image = cv2.bitwise_and(gray, binary)
    
  # find lines
  (lx,ly,lfx,lfy) = leftLine.findLine(masked_image,True)
  (rx,ry,rfx,rfy) = rightLine.findLine(masked_image,False)

  #for debug show binary and masked image plus found points 
  #showBinaryAndWarped(binary,masked_image,lx,ly,rx,ry,lfx,lfy,rfx,rfy)

  #smooth the lines using a history
  lfxs,lfys = leftLine.getSmoothed()
  rfxs,rfys = rightLine.getSmoothed()
   
    
  if len(lfys) >=2 and len(rfys) >=2:     
    centerLine = createCenterLine(image,leftLine,rightLine)
    image = centerLine.drawInfoText(image)
    
    result = drawLinesOnRoad(image,Minv,masked_image,lfxs,lfys,rfxs,rfys)
                                 
  else:
    result = image
  
  if debug:
    debug= showDebug(masked_image,lx,ly,rx,ry,lfxs,lfys,rfxs,rfys)    
    if not imageName is None:
      path_to_image = os.path.join("output_images","{0}_{1}.jpg".format(imageName,"undist"))
      scipy.misc.imsave(path_to_image, undist)
      path_to_image = os.path.join("output_images","{0}_{1}.jpg".format(imageName,"warped"))
      scipy.misc.imsave(path_to_image, warped)
      path_to_image = os.path.join("output_images","{0}_{1}.jpg".format(imageName,"binary"))
      scipy.misc.imsave(path_to_image, binary)
      path_to_image = os.path.join("output_images","{0}_{1}.jpg".format(imageName,"birdeye"))
      scipy.misc.imsave(path_to_image, masked_image)
      path_to_image = os.path.join("output_images","{0}_{1}.jpg".format(imageName,"result"))
      scipy.misc.imsave(path_to_image, result)    
      path_to_image = os.path.join("output_images","{0}_{1}.jpg".format(imageName,"debug"))
      scipy.misc.imsave(path_to_image, debug)
    result = np.concatenate((result,debug),axis=1)
    
  return result

global image_counter,leftLine,rightLine
leftLine = line.Line() 
rightLine = line.Line() 
image_counter=0

resultPath = "output_images/"
INPUT_DIR = "test_images"
if __name__ == '__main__':
  print(os.listdir(INPUT_DIR))

  #load calibration
  (mtx,dist,image_shape) = cb.getCalibration()

  (M,Minv,image_shape) = cp.getPerspective()

  #loop over the test pictures
#  if True:
#    item = "test4.jpg"
  for item in os.listdir("test_images/"):
    #Read in and grayscale the image
    print(item)
    path_to_image = os.path.join("test_images",item)
    image = mpimg.imread(path_to_image)    

    result = process_colored_image(image,image_shape,mtx,dist,M,Minv,False,True,item[:-4])

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=50)

    ax2.set_title('Processed Image', fontsize=50)
    ax2.imshow(result)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    path_to_result = os.path.join(resultPath,"result_{0}".format(item))
    fig = plt.gcf()
    fig.savefig(path_to_result) 
    plt.show()

