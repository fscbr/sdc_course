#class to create and average a heat map 
class Heatmap():
  def __init__(self,threshold):
    self.recent_maps = []
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
    
  def average(self):
    if len(self.recent_maps) > 1:
      maps = np.array(self.recent_maps)
      self.map = np.mean(maps,axis=0)
        
    
  #answer the averaged heat map
  def getMap(self):
    return self.map

  #draw the heatmap boxes
  def drawLabeledBoxes(self,img,color,drawHeatImage):
    map = self.map.copy()
    map[map <= self.threshold] = 0
    labels = label(map)
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
        
        # Define a bounding box based on min/max x and y
#        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bbox = ((xc-width, yc-height), (xc+width, yc+height))
        
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], color, 6)
    # Return the image
    if drawHeatImage:
      heatImage = cv2.cvtColor(np.array(self.map*100).astype(np.uint8), cv2.COLOR_GRAY2RGB)
      heatImage[:,:,1]=0
      heatImage[:,:,2]=0
      img = cv2.addWeighted(img, 1, heatImage, 0.5, 0)
    return img

  #create search windows using the known car positions
  def getSearchAreas(self,img):
    boxes = []
    if self.map is None:
      return boxes

    labels = label(self.map)
    # Iterate through all detected cars
    xmax = img.shape[1]
    ymax = img.shape[0]
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        x1 = np.min(nonzerox)
        x2 = np.max(nonzerox)
        y1 = np.min(nonzeroy)
        y2 = np.max(nonzeroy)
        
        size = max(x2-x1,y2-y1)
        
        #get the center
        xc = int((x2 + x1)/2)
        yc = int((y2 + y1)/2)
        #create boxes in three sizes
        l = [(64,86,64,0.66),(96,96,96,0.66),(129,129,128,0.66)]
    
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
            
          boxes += slide_window(img, (xc-dx,xc+dx), (yc-dy,yc+dy),(s, s),(f, f))       
    return boxes
    