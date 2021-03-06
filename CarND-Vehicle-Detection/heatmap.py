#class to create and average a heat map 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage.measurements import label
import featureDetection as fd

class Heatmap():
  def __init__(self,threshold):
    self.recent_maps = []
    self.cars = []
    self.lostCars = {}
    self.forecast = []
    self.map = None
    self.threshold = threshold
    
  #add boxes to the current heat map, threshold it and average it over the history 
  def update(self,shape,bboxes):
    self.map = np.zeros(shape)
    
    # Iterate through list of bboxes
    for box in bboxes:
      # Assuming each "box" takes the form ((x1, y1), (x2, y2))
      self.map[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    
    if len(self.recent_maps) > 10:
      self.recent_maps.pop()
    self.recent_maps.insert(0,self.map.copy())

  #calculate the mean over the history of heat maps    
  def average(self):
    if len(self.recent_maps) > 1:
      maps = np.array(self.recent_maps)
#      self.map = np.sum(maps,axis=0)
      self.map = np.mean(maps,axis=0)
    
  #answer the averaged heat map
  def getMap(self):
    return self.map
    
  #calculate the car boxes 
  def calcBoxes(self,img):
    map = self.map.copy()
    map[map <= self.threshold] = 0
    labels = label(map)
    
    cars = []
    self.forecast = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
         # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        xmin = np.min(nonzerox)
        xmax = np.max(nonzerox)
        ymin = np.min(nonzeroy)
        ymax = np.max(nonzeroy)
        #get center
        xc = int((xmin + xmax)/2)
        yc = int((ymin + ymax)/2)
        
        width =  int(max(96,xmax - xmin)/2)
        height =  int(max(96,ymax - ymin)/2)
        
        xy = (xc,yc,width,height)
        cars.append(xy)
        
        mxy = self.getMovement(xy)
        (mx,my) = mxy
        
#        print(xc,yc,"->",mx,my)     
        #forecast the next postion
        self.forecast.append((xc+mx,yc+my))

    #merge car positions     
    self.mergeCars(cars)
    
  #draw the heatmap boxes and the heatmap if requested
  def drawLabeledBoxes(self,img,color,drawHeatImage):
    # Iterate through all detected cars
    for car in self.cars:
        (xc,yc,width,height) = car
        # Define a bounding box based on min/max x and y
        bbox = ((xc-width, yc-height), (xc+width, yc+height))
        
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], color, 6)
    # Return the image
    if drawHeatImage:
      heatImage = cv2.cvtColor(np.array(self.map*10).astype(np.uint8), cv2.COLOR_GRAY2RGB)
      heatImage[:,:,1]=0
      heatImage[:,:,2]=0
      img = cv2.addWeighted(img, 1, heatImage, 0.5, 0)
    return img

  #create search windows using the known car positions
  def getSearchAreas(self,img):
    boxes = []
    # Iterate through all detected cars
    xmax = img.shape[1]
    ymax = img.shape[0]
    for car in self.forecast:
        (xc,yc) = car
        #create boxes in three sizes
        l = [(42,48,64,0.9),(58,72,96,0.9),(77,96,128,0.9)]
    
        for i in range(len(l)):
          (dy,dx,s,f) = l[i]
          if xc + dx > xmax:
            xc = xmax - dx
            
          if xc - dx < 0:
            xc = dx 
            
          if yc - dy < 390:
            yc = 390 + dy
            
          if yc + dy > ymax:
            yc = ymax - dy
            
          if yc - dy < 0:
            yc = dy  
            
          boxes += fd.slideWindow(img, (xc-dx,xc+dx), (yc-dy,yc+dy),(s, s),(f, f))       
    return boxes

  #answer movement vector if a vector close to the parameter has been found     
  def getMovement(self,xy):
    (xc,yc,width,height) = xy

    bestxy = self.getClosestPosition(xy)
      
    if not bestxy is None:
      (hxc,hyc,width,height) = bestxy 
#      print("movement: ",xc-hxc,yc-hyc)
      return (xc-hxc,yc-hyc)
    return (0,0)

  #merge the old and the new car position list to avoid detection losses from one to the other image
  # not found cars are ignored after 10 images
  def mergeCars(self,cars):
    old_cars = self.cars
    self.cars = cars
    for xy in old_cars:
      if self.getClosestPosition(xy) is None:
        counter = self.lostCars.get(xy,None)
        if counter is None:
          self.lostCars[xy] = 1
        elif counter < 10: 
          self.lostCars[xy] = counter+1
        else:
          del self.lostCars[xy]

  #find the closest position in the car list related to a position. Return None if no close position is found.
  def getClosestPosition(self, xy):
    (xc,yc,width,height) = xy
    bestdistance = 999
    bestxy = None
    for hxy in self.cars:
      (hxc,hyc,hwidth,hheight) = hxy
      distance = (xc-hxc)*(xc-hxc) + (yc-hyc)*(yc-hyc)

      if distance < bestdistance:
        bestxy = hxy
        bestdistance = distance
      
    if not bestxy is None:
      (hxc,hyc,width,height) = bestxy 
      return (bestxy)
#    print("not found: ",xy,bestdistance)
    return None

