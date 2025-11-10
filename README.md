# Unity-Rover-Object-Detection
This repo is not meant to be an official once, simply a REPO for me and Siya to use for file sharing while we work through YOLO. Once a full training or testing script has been written, it will be added onto the main REPO with CI and formatting.
## Setup
*Requires Python 3*
### Package Installation
install Ultralytics using

    pip install ultralytics

install OpenCV using 

    pip install opencv-python

### To Use Webcam Streaming
    python3 test_webcam.py

### To train and validate
Use a custom dataset, and set the .yaml path in the 'train_yolo.py' file

    python3 train_yolo.py

