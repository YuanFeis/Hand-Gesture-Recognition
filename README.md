# Hand-Gesture-Recognition

Automatic detection and classification of dynamic hand gestures in real-world environments and its application in human-computer interfaces is a challenging task that has gained significant scientific interest in recent years. 

This project aims to implement a system that performs simultaneous detection and classification of dynamic hand gestures from multimodal data and further, uses the detected gestures to control desktop applications. The dataset used is the [Jester](https://20bn.com/datasets/jester) dataset which contains 148K short clips of videos that depict a person performing a gesture in front of a camera.

## Instructions

### Setting up the system
1. Fork this repository and clone the forked repository.
2. Change directory using `cd Hand-Gesture-Recognition`.
3. Create a Python 3.7 virtual environment using conda/pipenv/virtualenv and activate it.
4. Use `pip install -r requirements.txt` to install the requirements.

### Running the gesture control demo using pretrained model
1. Change to the `gesture-control` directory using `cd gesture-control`.
2. Run `python hanuman.py --config "gesture_control_config.cfg"`.
3. Wait for the program to load. Open a YouTube video in your browser and click anywhere on the page to focus.
4. Now, perform any of the gestures specified in the mapping.ini file to control YouTube.
