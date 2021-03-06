import numpy as np
import scipy.misc
import math
import cv2

class Line():
  def __init__(self):
    # was the line detected in the last iteration?
    self.detected = False  
    self.detected_in_line_area = False  
    # x values of the last n fits of the line
    self.recent_xfitted = []     
    #radius of curvature of the line in some units
    self.radius_of_curvature = 0.0     
    #distance in meters of vehicle center from the line
    self.offset = 0.0
    #fitted parameter
    self.fit = None
    
  def create(self,lx,ly):
    (lfx,lfy,lfit) = self.fitSecondOrderPolynom(lx,ly)
    self.update(lfit,lfx,lfy)
    
  #update a line. add fit topx and y to the stack, get the mean values over the stack
  def update(self, fit, x, y):
    if len(self.recent_xfitted) > 5:
      self.recent_xfitted.pop()
    self.recent_xfitted.insert(0,fit)      
    
    fit = np.array(self.recent_xfitted)
    #averaged fit parameter
    self.fit = np.mean(fit,axis=0)
    
#    print(len(self.recent_xfitted))
    
  def calcX(self,y):
    xf = self.fit[0]*y**2 + self.fit[1]*y + self.fit[2]
    return np.array(xf).astype("int")
    
  def getSmoothed(self):
    y = np.array(np.linspace(70, 720, num=20)).astype("int")
    x = self.calcX(y)
    return (x,y)
        
  def calcOffset(self,image):
    x = self.calcX(np.array(image.shape[0]))
    self.offset = (image.shape[1] / 2 - x)*3.7/540
    
  def calcCurveRad(self):  
# Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/550 # meters per pixel in y dimension extracted from left dashed line in the transformed image (getCameraPerspective)
    xm_per_pix = 3.7/540 # meteres per pixel in x dimension extracted from distance between both lines in the transformed image (getCameraPerspective)

    y = np.array(np.linspace(0, 720, num=10))
    x = self.calcX(y)
    y_eval = np.max(y)

    fit_cr = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
    self.radius_of_curvature  = ((1 + (2 * fit_cr[0] * y_eval + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])
           
#Draws information about the center offset and the current lane curvature onto the given image.
  def drawInfoText(self,image):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'Radius of Curvature = %d(m)' % self.radius_of_curvature, (50, 50), font, 1, (255, 255, 255), 2)
    left_or_right = 'left' if self.offset < 0 else 'right'
    cv2.putText(image, 'Vehicle is %.2fm %s of center' % (np.abs(self.offset), left_or_right), (50, 100), font, 1, (255, 255, 255), 2)
    return image

  def findLine(self,image,isLeft):    
    lfx = []
    lfy = []
 
    if not self.fit is None:
      #find peaks in close to line history
      (lx,ly) = self.findPeak(image,isLeft,self.fit)      
    
      self.detected_in_line_area = len(lx >= 3)
      if self.detected_in_line_area:
        (lfx,lfy,lfit) = self.fitSecondOrderPolynom(lx,ly)
    
      self.detected = self.detected_in_line_area
        
    if not self.detected_in_line_area:
      (lx,ly) = self.findPeak(image,isLeft)
      self.detected = len(lx >= 3)
      if self.detected:
        (lfx,lfy,lfit) = self.fitSecondOrderPolynom(lx,ly)
        
    if self.detected:
      dx = lfx - lx
      lx2,ly2 = self.removeOutlier(dx,lx,ly,10)
      (lfx,lfy,lfit) = self.fitSecondOrderPolynom(lx2,ly2)
    
      self.update(lfit,lfx,lfy)
      return (lx2,ly2,lfx,lfy)
    return (lfx,lfy,lfx,lfy)

 #Removes horizontal outliers based on a given percentile.
  def removeOutlier(self,dx,x, y, q=5):
    if len(x) < 3 or len(y) < 3:
        return x, y

    dx = np.array(dx)
    x = np.array(x)
    y = np.array(y)

    lower_bound = np.percentile(dx, q)
    upper_bound = np.percentile(dx, 100 - q)
#  print(lower_bound ,upper_bound)
    selection = (dx >= lower_bound) & (dx <= upper_bound)

    x = x[selection]
    y = y[selection]
    return x,y

  #find peaks in the left or right half of an image. If a fit paramter is set, the peak is search in an area around a spline line
  def findPeak(self,img,isLeft,fit=None):
    
    width = img.shape[1]
    height = img.shape[0]
    
    wh = int(width/2)
    hh = int(height/2)
    
    aoi_x = int(width/6)
    if not isLeft:
      aoi_x = wh
    
    aoi_w = int(width/3)
    
    lr = range(aoi_x,aoi_w+aoi_x)
    image = img[:,aoi_x:aoi_w+aoi_x]
    ly = []
    lx = []
    #find first peak in the lower 200 pixel
    if fit is None:
      histogram = np.sum(image[image.shape[0]-200:image.shape[0],:], axis=0)      
      lc = int(np.argmax(histogram))    
      offsetx = 70
    
    step_width = 10
    steps = int(height / 10) 
    for i in range(steps):
      to = height - int(step_width *i)
      f = height- int(step_width *(i+1))

      #get expected line x using history
      if not fit is None:        
        lc = fit[0]*to**2 + fit[1]*to + fit[2] - aoi_x
        offsetx = 40
    
      if lc >= aoi_w or lc < 0:
        break;
    
      histogram = np.sum(image[f:to,:], axis=0) 
      hi = histogram < 10
      histogram[hi] = 0

      peak = int(np.argmax(histogram[max(0,lc-offsetx):min(lc+offsetx,aoi_w)]))
    
      y = int((f+to)/2)
      if peak > 0:
        x=max(min(peak+lc-offsetx,aoi_w),0)
                
        ly.append(y)
        lx.append(x)
        lc = x
                
    lx = np.array(lx) + aoi_x

    return (np.array(lx).astype("int"),np.array(ly).astype("int"))

  
  def fitSecondOrderPolynom(self,x,y):
    fit = np.polyfit(y, x, 2)

    fitx = fit[0]*y**2 + fit[1]*y + fit[2]
    return (np.array(fitx).astype(int),np.array(y).astype(int),fit)
    

