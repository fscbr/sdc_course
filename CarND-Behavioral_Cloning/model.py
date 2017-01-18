import os
import numpy as np
from keras.models import Model,Sequential
from keras.layers import Dense, Dropout, Flatten, Activation,Input, Convolution2D, MaxPooling2D
from keras.models import model_from_json
from keras.callbacks import EarlyStopping,ProgbarLogger,ModelCheckpoint
from keras.optimizers import Adam
import cv2
import pandas as pd
import preprocess as pp

def get_model(row,col,ch):
  img_input = Input((row, col,ch))
  print(img_input.get_shape())
  x = Convolution2D(32, 3, 3, activation='relu', border_mode='same',init='normal')(img_input)
  x = MaxPooling2D((2, 2), strides=(2, 2))(x)
  x = Dropout(0.15)(x)

  x = Convolution2D(64, 3, 3, activation='relu', border_mode='same',init='normal')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2))(x)
  x = Dropout(0.15)(x)

  x = Convolution2D(64, 3, 3, activation='relu', border_mode='same',init='normal')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2))(x)
  x = Dropout(0.15)(x)

  x = Flatten()(x)
  x = Dense(1024, activation='relu',init='normal')(x)
  x = Dropout(0.15)(x)
  x = Dense(256, activation='relu',init='normal')(x)
  x = Dense(64, activation='relu',init='normal')(x)
  y = Dense(1)(x)

  model = Model(input=img_input, output=y)
  model.compile(optimizer="nadam", loss="mse")
  return model

def get_augmented_row(row):
  steering = row["steering"]
  # randomly choose the camera to take the image from
  camera = np.random.choice(['center', 'center', 'center', 'left','right'])

  image_file = os.path.join(data_dir,"{0}".format(row[camera].strip()))
  image = cv2.imread(image_file,cv2.IMREAD_COLOR )
  # adjust the steering angle for left and right cameras
  if camera == "left":
    steering += 0.25
  elif camera == "right":
    steering -= 0.25
  
  # decide whether to horizontally flip the image:
  # This is done to reduce the bias for turning left that is present in the training data
  flip_prob = np.random.random()
  if flip_prob > 0.5:
    # flip the image and reverse the steering angle
    steering = -1*steering
    image = cv2.flip(image,1)

  image = pp.trans_image(image,40)
  # Crop, resize and normalize the image
  image = pp.preprocess_image(image)
  image = image.reshape(image.shape[0],image.shape[1],3)
  return image, steering


def get_data_generator(data_frame, batch_size=32):
  N = data_frame.shape[0]
  batches_per_epoch = N // batch_size

  print("data frame size:",data_frame.shape[0]," batches_per_epoch:",batches_per_epoch)

  i = 0
  while(True):
    start = i*batch_size
    end = start+batch_size - 1

    X_batch = np.zeros((batch_size, 64, 64, 3), dtype=np.float32)
    y_batch = np.zeros((batch_size,), dtype=np.float32)

    j = 0

    # slice a `batch_size` sized chunk from the dataframe
    # and generate augmented data for each row in the chunk on the fly
    for index, row in data_frame.loc[start:end].iterrows():
      X_batch[j], y_batch[j] = get_augmented_row(row)
      j += 1

    i += 1
    if i == batches_per_epoch - 1:
      # reset the index so that we can cycle over the data_frame again
      i = 0
    yield X_batch, y_batch



if __name__ == "__main__":
# Parameters
  batch_size = 32
  nb_epoch = 16
  model_name = "model"
  data_dir = "../CarND-Behavioral_Cloning-work/data/merged"
  model_dir = "./"
  np.random.seed(1000)

#read data file
  data_file = os.path.join(data_dir, "driving_log.csv")
  data_frame = pd.read_csv(data_file, usecols=[0, 1, 2, 3])

#reduce 0 steering in the data to improve distribution of steering for the training
# seperate 0.0 steering and others
  data_frame1 = data_frame.query("steering != 0.0")
  data_frame2 = data_frame.query("steering == 0.0")
#drop 85% of zero steering
  data_frame2 = data_frame2.sample(frac=.15).reset_index(drop=True)
  data_frame = pd.concat([data_frame1,data_frame2])

# shuffle the data
  data_frame = data_frame.sample(frac=1).reset_index(drop=True)

# 80-20 training validation split
  training_split = 0.8
  num_rows_training = int(data_frame.shape[0]*training_split)
  training_data = data_frame.loc[0:num_rows_training-1]
  validation_data = data_frame.loc[num_rows_training:]

#smples per epoch a 6 times larger than the data set length (3 cameras and flip of image)
  samples_per_epoch=int(len(training_data)/batch_size+1)*batch_size*10
  nb_val_samples=int(len(validation_data)/batch_size)*batch_size*10

  print("samples_per_epoch:",samples_per_epoch," nb_val_samples:",nb_val_samples)

# release the main data_frame from memory
  data_frame = None

# use the a data generator to get the image data on the fly
  training_generator = get_data_generator(training_data, batch_size=batch_size)
  validation_data_generator = get_data_generator(validation_data, batch_size=batch_size)

#create the model
  model = get_model(64,64,3)
#  model = get_model2()
  print("model created")

#save the model configuration
  model_file = os.path.join(model_dir, "{0}.json".format(model_name))
  with open( model_file, 'w') as model_file:
    model_file.write(model.to_json())
  print("model saved")

#create callbacks for save of model weights after an epoch, log and early stopping
  weights_file = os.path.join(model_dir, "{0}_weights_{1}.hdf5".format(model_name,"{epoch:02d}"))
  checkcb = ModelCheckpoint(weights_file, verbose=0, mode='auto')
  earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2, mode='auto')
  progbar = ProgbarLogger()

#train the model
  model.fit_generator(training_generator, validation_data=validation_data_generator, samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch, callbacks=[progbar,earlyStopping,checkcb], nb_val_samples=nb_val_samples)


