import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO


from keras.models import load_model
import h5py
from keras import __version__ as keras_version

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None



class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral
        
        
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


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)
        steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))

        # added ability for controller to adjust desired speed during a turn
        controller.adjust_speed(float(speed), steering_angle)
       
        throttle = controller.update(float(speed))

        send_control(steering_angle, throttle)
        print(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    throttle = controller.update(0.0)
    send_control(0, throttle)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()
    
    # check that model Keras version is same as local Keras version
    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)
              
    model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
