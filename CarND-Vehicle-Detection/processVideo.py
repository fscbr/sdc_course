# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import cv2
import numpy as np
import processChain as pc
import heatmap as hm
import trainSvm as ts
import time
import globalData as gd

def process_image(image):
  return pc.processColoredImage(image,True,True,None,X_scaler,svc,param)
    
#load trained model
data = {}
data = ts.getModelData()
svc = data["svc"]
X_scaler = data["X_scaler"]
svc = data["svc"]
param = data["param"]

#init global data
gd.data["image_counter"]=0
gd.data["heatmap"]=hm.Heatmap(1)

#process video
white_output = 'project_result.mp4'
clip1 = VideoFileClip("project_video.mp4")
#white_output = 'test_result.mp4'
#clip1 = VideoFileClip("test_video.mp4")
white_clip = clip1.fl_image(process_image)
#%time 
white_clip.write_videofile(white_output, audio=False)   