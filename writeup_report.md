12-MAR-2021
# Behavioral Cloning
This is a Udacity Self-Driving Car NanoDegree project submission that uses deep learning to clone driving behavior.

![](./wup_assets/mtn-turning.png)

## Table of Contents
- [**Required Files**](#required-files)
- [**Quality of Code**](#quality-of-code)
  - [Code Functionality](#code-functionality)
  - [Code Usability](#code-usability)
  - [Code Readability](#code-readability) 
- [**Model Architecture and Training Strategy**](#model-architecture-and-training-strategy)
  - [General Architecture Employed](#general-architecture-employed)
  - [Reducing Overfitting](#reducing-overfitting)
  - [Parameter Tuning](#parameter-tuning)
  - [Training Data Chosen](#training-data-chosen)
- [**Architecture and Training Documentation**](#architecture-and-training-documentation)  
  - [Solution Design Approach](#solution-design-approach)
  - [Final Model Architecture](#final-model-architecture)
  - [Training Data Set](#training-data-set)
- [**Simulation**](#Simulation)    

## Required Files
- `model.py`: Python script used to create and train the model.
- `drive.py`: Python script used to autonomously drive the car. This script has the following modifications:
  - Desired speed was increased to 30 mph from the original 9 mph.
  - Vehicle will slow down aggressively when initiating a turn, then returns to desired speed.
- `model.h5` : The saved model.
- `writeup_report.md`: writeup of project for Udacity evaluator.
- `video.mp4`: video recording of vehicle driving around track 2 in the opposite direction from the driver's perspective. Track 2 was chosen for the recording because there were more exciting turns. The third person perspective of this video can be viewed in `auto_hard_rev_3p.mp4`.

## Quality of Code

### Code Functionality

The model was trained with data from recording 2 laps of driving on track 1 (easy lakeside road) and track 2 (difficult mountainous road). The model was saved to `model.h5` and can be observed to successuly operate the simulation in track 1 and track 2 using the following:

`python drive.py model.h5`

### Code Usability 

The script to create the model is `model.py`. It uses a data generator class, `DrivingLogSequence`, that inherits from `keras.utils.Sequence`. This class accesses data from two folders, `data_easy_route` and `data_hard_route`, to generate images and steering angle data for training rather than storing the entire folders of images and steerings angles into memory (the data in those folders are compressed `in data_easy_route_7z` and `in data_hard_route_7z` folders, and must be uncompressed into the base directory to use in training).

The main function in `model.py` that creates the neural network is `create_model()`. The function `train_model()` then trains the model using `DrivingLogSequence` data generators and saves the model to the file `model.h5`

### Code Readability

The code follows [PEP-8](https://www.python.org/dev/peps/pep-0008/) Style Guide as much as possible and [PEP-257](https://www.python.org/dev/peps/pep-0257/) Docstring Conventions. For instance:

```python
# ...etc...
    def samples(self, camera, img, steering):
        '''
        Creates a list of image and steering data for 
        adding to a batch of samples. Used in __getitem__().
        
        Params
            camera: name of camera; 'center', 'left', or 'right'.
            img: rgb image of camera
            steering: steering ANGLE (float) applied
        '''
# ...etc...
```

["Snake case"](https://en.wikipedia.org/wiki/Snake_case) is predominantly used except for class names which use ["Camel Case"](https://en.wikipedia.org/wiki/Camel_case). For instance:

```python
# ...etc...
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
# ...etc...
```

## Model Architecture and Training Strategy

### General Architecture Employed

The model architecture is a linear stack of layers and is subclassed from `keras.Sequential`. It preprocesses the image with normalization and cropping, then employs convolution layers followed by several dense layers to output a steering angle.

### Reducing Overfitting

Rectified Linear Units served as the activation functions for each layer which was followed by dropout to reduce overfitting. Adding the non-linearities and dropout helped seemed to have the following effect:

- Decrease the final training loss from a mean squared error of around 0.06 to 0.02. 
- The model demonstrated an ability to generalize by driving in the opposite direction around both tracks using only training data of driving in the default direction.

### Parameter Tuning

The model uses an adam optimizer so a learning rate parameter was not chosen. This was sufficient in training the vechicle to drive on both track 1 and 2 succesffully. 

### Training Data Chosen

The training data was initially the default data provided by Udacity. However, the resulting model underperformed when driving on both tracks. Ultimately, default data provided by Udacity was discarded and replaced with data from the video recording of driving two laps around tracks 1 and 2 with their associated steering angles. The training data is contained in the following:

- `data_easy_route`: folder data from driving on track 1 (must be uncompressed first from folder data_easy_route_7z to use)
  - `driving_log.csv`: comma separated value of image filenames and associated steering angles
  - `IMG`: folder containing images from center, left and right cameras of vehicles
- `data_hard_route`: folder data from driving on track 2 ((must be uncompressed first from folder data_hard_route_7z to use)
  - `driving_log.csv`: comma separated value of image filenames and associated steering angles
  - `IMG`: folder containing images from center, left and right cameras of vehicles
  
## Architecture and Training Documentation

### Solution Design Approach

The approach to designing the model simply to recreate the model described in _"End-to-end Learning for Self-Driving Cars" by Bojarski et. al., 25APR2016_, then iteratively improve performance (lower training loss) by experimenting with the following:

- Adding activation function 
- Adding dropout 
- Initialy using default Udacity training dataset
- Augmenting default training dataset with manipulated camera images and steering inputs
- Creating own training datasets of track 1 and track 2 driving


### Final Model Architecture

After multiple iterations of modifying the model and training data, the final architecture is depicted below:

![](./wup_assets/cnn_architecture.png)

It is similar to _Bojarski et. al._'s model and has the following features:
- Preprocessing with normalization and cropping. Normalizing the image will help in finding the appropriate model weights and biases, and croppping the image removes unecessary data like the images of the surrounding landscape.
- Convolutional layers with `relu` activations (but same filter, kernel, and stride parameters as _Bojarski et. al._'s model).
- Extensive use of dropout after almost every layer except the last. Empirically, it seemed that using a low dropout rate of 0.1 or 0.2 helped in reducing overfitting than using a dropout value of 0.5 after the convolutions.

Below is a portion of the function `create_model()` which creates the final model:

```python
# ...etc...
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
# ...etc...
```

###  Training Data Set

The default Udacity dataset was initially used which consisted of images from cameras mounted on the left, center and right of the vehicle, and a correspoding `steering_angle` for the center camera. 

To minimize use of memeory, a data generator, `DrivingLogSequence`, subclassed from `keras.utils.Sequence`, was used to fetch the images and steering_angle. 

The original images were flipped horizontally to provide additional data for driving in the opposite direction of the track. The corresponding steering angle for the flipped image is simply the negative of the original, `flipped_steering_angle = -steering_angle`.

![](./wup_assets/drv_udacity_data_image_1.png)

The images from the left and right cameras were also shifted further left or right respectively, to provide additional steering correction data. Through trial and error, a 50-pixel shift seemed suitable.

![](./wup_assets/drv_udacity_data_image_shift_right.png)
![](./wup_assets/drv_udacity_data_image_shift_left.png)

A angle of `0.35` seemed suitable and was applied to `steering_angle` for the left and right cameras (the correction angle was subtracted from the `steering_angle` for the right camera). 

```python
# correction ANGLEs for each camera
ANGLE       = 0.35
CORR_ANGLES = {'center': 0, 'left': ANGLE, 'right': -ANGLE}
```
The correction angle was applied again to the steering angle for the images that were shifted left and right.

Below is a portion of the code that, in the method `__getitem__()` reads the original image and `steering_angle` (with `CORR_ANGLES` applied), and then generates, in the method `samples()`, the shifted image and a shifted `steering_angle`.

```python
class DrivingLogSequence(Sequence):
    
# ...etc...

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
    
# ...etc...
```

In the final training runs, flipping the image horizontally was discarded. This is because:

- It seemed to cause the vehicle to swerve left and right on straightaways. Removing the flipped image seemed to reduce the frequent left and right corrections on straightaways, especially on the bridges of track 1 and track 2.
- In track 2, after a tight right turn causes the vehicle to drift into the left lane, the vehicle did not correct back to the right lane until a tight left turn. This is likely because the flipped horizontal image on track 2 makes the vehicle appear to be in the left lane, and so it is trained that the left lane is a suitable lane for driving and will not correct back to the right. Removing the flipped image enabled the vehicle to correct back to the right lane on track 2 if at the end of a tight right turn the vehicle ended up on the left lane.

![](./wup_assets/drv_flipped_left_image.png)

The default Udacity dataset was discarded after the vehicle underperformed. Instead, it was replaced with training data generated from recording two laps of driving around tracks 1 and 2 (`data_easy_route` and `data_hard_route` respectively) in the default directions. Below is the history of the final training session that produced `model.h5`:

![](./wup_assets/history.png)

There was sufficient data that the model was able to generalize the training and successfully autonomously drive the vehicle in __both directions__ in both tracks, and in the case of track 2, maintain right lane driving or correct back to it.

## Simulation

`drive.py` was modified in the following ways:

- Desired speed was increased to 30 mph. The original speed of 9mph was agonizingly slow during test and evaluation.

- A simple cornering algorithm was used. Because the desired speed was increased to 30 mph, `drive.py` will aggressively slow down the vehicle when the steering angle is above a threshold value of 0.3, and will then attempt to maintain a turning speed of 23 mph throughout the turn. Once the turn is complete, it will resume the target speed of 30 mph.

- The class `RidgeRacerController`, subclassed from `SimplePIController`, was created to achieve the desired speed of 30 mph and slow down during turns. At no point does it modify the model predicted `steering_angle`.

```python
#  ...from drive.py:

class RidgeRacerController (SimplePIController):
    '''
    Controller for making vehicle go faster (30mph).
    With a higher speed, controller sets a more agressive throttle (Kp=2.0)
    and also adjusts speed when turning.
    '''
    def __init__(self):
        
        # Kp is 2.0 for more agressive throttle
        # either to brake or accelerate to target speed
        SimplePIController.__init__(self, Kp=2.0, Ki=0.002)
        
        # set RidgeRacer to target speed of 30 mph
        self.target_speed = 30
        self.turning = False
        self.prev_steering_angle = 0.0
        
        # steering angle threshold for commencing turn
        self.steering_angle_thresh = 0.3
        
        # speed to start braking if entering turn
        self.turn_speed = 23
        
        # speed to brake to when entering turn
        self.enter_turn_speed = 2.0
        
        # use a slow launch speed at start
        self.set_desired(0.3*self.target_speed)
        
    def adjust_speed (self, speed, steering_angle):
        '''
        Adjust vehicle speed to safely negotiate a turn.
        '''
    
        if not self.turning:
            if abs(steering_angle) > self.steering_angle_thresh:
                self.turning = True
                if speed > self.turn_speed:
                    # Entering Turn, slow down
                    self.set_desired(self.enter_turn_speed)
            else:
                # Going straight, setting target speed
                self.set_desired(self.target_speed)
        elif abs(steering_angle) > self.steering_angle_thresh:
            if self.set_point < self.turn_speed:
                # Still Turning, increasing to turn speed            
                self.set_desired(self.turn_speed)
        else:
            # Exiting turn, resume target speed        
            self.turning = False
            self.set_desired(self.target_speed)
            
        self.prev_steering_angle = steering_angle
        
        
# original controller spec:
# controller = SimplePIController(0.1, 0.002)

# use the RidgeRacer controller to go fast
controller = RidgeRacerController()

# ...etc...

```

Because the drive around the lake was boring, the video of the vehicle driving in the opposite direction around track 2 was used for the submission video, `video.mp4`. Additional videos of the vehicle driving autonomously around the track are as follows: 

- From 3rd person perspective:
  - `auto_easy_3p.mp4`: driving around track 1.
  - `auto_easy_rev_3p.mp4`: driving in opposite direction around track 1.
  - `auto_hard_3p.mp4`: driving around track 2.
  - `auto_hard_rev_3p.mp4`: driving in opposite direction around track 2.  
- From driver perspective:
  - `auto_easy.mp4`: driving around track 1.
  - `auto_easy_rev.mp4`: driving in opposite direction around track 1.
  - `auto_hard.mp4`: driving around track 2.

![](./wup_assets/drv_track_2_opp.png)