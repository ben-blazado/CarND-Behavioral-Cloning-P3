# Behavioral Cloning
This is a Udacity Self-Driving Car NanoDegree project submission that uses Machine Learning to teach a simulated vehicle how to autonomously apply steering commands.

![](./wup_assets/2021_03_11_00_59_53_234.jpg)

## Installation
Clone or fork this repository.

## Dependencies
- See `env-bcl-gpu.yml` for packages.
- [Udacity's Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim)

## Usage
Intended user is the Udacity evaluator for this project. To observe the vehicle autonomously driving, follow the following:

1. Start the autonomous driver: `python drive.py model.h5`
2. Start Udacity's Self-Driving Car Simulator (SDCS).
3. Select Track in the SDCS.
4. Select Simulation Mode in the SDCS. This connects the simulator to `drive.py` which drives the car around the selected track based on the machine learning model captured in `model.h5`.

## Files
### Project Files
- `model.py`: Python script used to create and train the model.
- `drive.py`: Python script used to autonomously drive the car.
- `model.h5` : The saved model.
- `writeup_report.md`: writeup of project for Udacity evaluator.
- `video.mp4`: video recording of vehicle driving around track 2.

### Other files 
- `env-bcl-gpu.yml`: Conda YAML file for installing dependencies
- `proto.ipynb`: Jupyter Notebook for prototyping python and markdown code
- Additional Videos:
  - `track_1_fwd.mp4`: video of vehicle driving around track 1.
  - `track_1_rev.mp4`: video of vehicle driving in opposite direction around track 1.
  - `track_2_rev.mp4`: video of vehicle driving in opposite direction around track 2.
- `training_log.csv`: CSV file showing training history of `model.h5`
- `video.py`: Udacity included script for generating video of vehicle driving
- 