# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import cv2
import numpy as np
import lineDetection as ld
import calibration as cb
import cameraPerspective as cp
import line

def process_image(image):
  global image_shape,mtx,dist,M,Minv
  return ld.process_colored_image(image,image_shape,mtx,dist,M,Minv,True,True,None)

global image_counter,leftLine,rightLine
global image_shape,mtx,dist,M,Minv
leftLine = line.Line() 
rightLine = line.Line() 
image_counter=0
white_output = 'project_result.mp4'

if __name__ == '__main__':
  global image_shape,mtx,dist,M,Minv

  #load calibration
  (mtx,dist,image_shape) = cb.getCalibration()
  #load perspective
  (M,Minv,image_shape) = cp.getPerspective()


  clip1 = VideoFileClip("project_video.mp4")
  white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
  white_clip.write_videofile(white_output, audio=False)

