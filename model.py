from math import ceil, floor
from random import shuffle
from keras.utils import Sequence

import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Lambda, BatchNormalization, Flatten, Dense
from keras.layers import Conv2D, Cropping2D, Dropout, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger


# correction ANGLEs for each camera
ANGLE       = 0.35
CORR_ANGLES = {'center': 0, 'left': ANGLE, 'right': -ANGLE}

# image shift values for left and right cameras
DX         = 50
SHIFT_VALS = {'left': DX, 'right': -DX}


def shift(img, DX):
    '''
    Non-destructively shifts img horizontally by DX pixels. .
    Used by DrivingLogSequence to create one more image further left or
    right of camera to help in providing addtional steering samples during 
    training when car is too close to edge of lane.
    
    Params
        img: img to shift
        DX:  number of pixels to shift. <0 shifts left, >0 shifts right
    '''
        
    shifted_img = img.copy()
    if DX > 0:
        # shift right
        shifted_img[:,DX:] = img[:,:-DX]
    else:
        # shift left
        shifted_img[:,:DX] = img[:,-DX:]

    return shifted_img


# ref: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DrivingLogSequence(Sequence):
    '''
    Generates driving log data for training. 
    Inherits from keras.utils.Sequence.
    Used in call to Model.fit_generator.
    
    '''

    def __init__ (self, driving_log, batch_size=16):
        '''
        Class initializer.
        
        Params:
            driving_log: a list of dicts containing the following keys:
                'center'   : filename to image from center camera
                'left'     : filename to image from left camera
                'right'    : filename to image from right camera
                'steering' : steering ANGLE applied
                
            batch_size: number of samples to get from driving log
        '''
        
        self.driving_log = driving_log
        self.batch_size = batch_size
        
        return
    
    def __len__(self):
        '''
        Returns total number of batches in driving log.
        '''
        return floor(len(self.driving_log)/self.batch_size)
    
    def samples(self, camera, img, steering):
        '''
        Creates a list of image and steering data for 
        adding to a batch of samples. Used in __getitem__().
        
        Params
            camera: name of camera; 'center', 'left', or 'right'.
            img: rgb image of camera
            steering: steering ANGLE (float) applied
        '''

        images    = []
        steerings = []
        
        # append base image
        images.append(img)
        steerings.append([steering])
        
        # if camera is left or right, augment data with image shifted
        # further left or right with additional steering correction
        if camera != 'center':
                # create shifted image and associated steering 
                shift_img       = shift(img, SHIFT_VALS[camera])
                shift_steering  = steering + CORR_ANGLES[camera]

                # append shifted image and associated steering
                images.append(shift_img)
                steerings.append([shift_steering])
                
        return images, steerings
        
    def __getitem__(self, index):
        '''
        Returns images and associated images from driving log during
        data generation.
        '''
        
        start = index
        end   = index + self.batch_size
        driving_log_batch = self.driving_log[start:end]
        
        X_images = []
        y_steerings = []
        
        for line in driving_log_batch:
            
            center_steering = float(line['steering'])
            
            for camera in CORR_ANGLES:
                
                img_filename = line[camera]
                
                img = plt.imread(img_filename)
                steering = center_steering + CORR_ANGLES[camera]
                
                images, steerings = self.samples(camera, img, steering)
                
                X_images.extend(images)
                y_steerings.extend(steerings)
            
        X = np.array(X_images)
        y = np.array(y_steerings)
        
        return X, y
    
    def on_epoch_end(self):
        shuffle(self.driving_log)
        return
    

def create_generators(test_size=0.2, shuffle=True):
    '''
    Creates generators for the training and validation sets.
    '''
    
    drv_log_folders = ['data_easy_route',
                       'data_hard_route']
    drv_log_filename = 'driving_log.csv'

    drv_log = []
    for drv_log_folder in drv_log_folders:
        with open(drv_log_folder + '/' + drv_log_filename) as driving_log_file:
            driving_log_reader = csv.DictReader(driving_log_file)
            for line in driving_log_reader:
                for camera in CORR_ANGLES:
                    line[camera] = drv_log_folder + "/" + line[camera].strip()
                drv_log.append(line)
                
    training, validation = train_test_split(drv_log, 
                                            test_size=0.2, 
                                            shuffle=True)                                       
                       
    return DrivingLogSequence(training), DrivingLogSequence(validation)        


def normalize(rgb):
    '''
    Normalizes rgb image between [-1, 1].
    Used in Lambda layer of model.
    '''
    return (rgb-128.0) / 128.0
    
    
def callbacks():
    '''
    Retuns list of early stopper and training logger for use in training.
    '''
    # https://keras.io/api/callbacks/early_stopping/
    early_stopper = EarlyStopping(monitor='val_loss', 
                                  patience=6, 
                                  restore_best_weights=True)
                                  
    training_logger = CSVLogger("training_log.csv")
    
    return [early_stopper, training_logger]
    
    
def create_model():
    '''
    Creates a model for use in autonmous driving mode. Architecture is 
    similar to the one in ref: Bojarski et. al., "End to end Larning 
    for Self-Driving Cars", 25APR2016.
    '''

    model = Sequential()
    model.add(Lambda(normalize, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=[(50, 20), (0, 0)]))
    
    model.add(Conv2D(filters=24, kernel_size=5, strides=2, activation='relu'))
    model.add(Dropout(0.1))
    
    model.add(Conv2D(filters=36, kernel_size=5, strides=2, activation='relu'))
    model.add(Dropout(0.1))
    
    model.add(Conv2D(filters=48, kernel_size=5, strides=2, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(10, activation='relu'))
    
    model.add(Dense(1))    
    
    model.compile(loss='MSE', optimizer='Adam')
    
    return model
    
    
def train_model(model):
    '''
    Trains the model and saves it to "model.h5" file.
    Uses DrivingLogSequence objects driving_log_seq_training and 
    driving_log_seq_validation to generate data for training.
    '''
    
    driving_log_seq_training, driving_log_seq_validation = create_generators()

    model.fit_generator(driving_log_seq_training, 
                        validation_data=driving_log_seq_validation, 
                        epochs=30,
                        callbacks=callbacks())
                        
    model.save('model.h5')
                        
    return


if __name__ == '__main__':
    model = create_model()
    train_model(model)
    
