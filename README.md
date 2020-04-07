# Hand-Gesture-Recognition

Automatic detection and classification of dynamic hand gestures in real-world environments and its application in human-computer interfaces is a challenging task that has gained significant scientific interest in recent years. 

This project aims to implement a system that performs simultaneous detection and classification of dynamic hand gestures from multimodal data and further, uses the detected gestures to control desktop applications. The dataset used is the [Jester](https://20bn.com/datasets/jester) dataset which contains 148K short clips of videos that depict a person performing a gesture in front of a camera.


## How it works?

### The dataset

We used the 20BN-JESTER dataset, which is a large collection of densely-labeled video clips that show humans performing pre-defined hand gestures in front of a laptop camera or webcam. It allows for training robust machine learning models to recognize human hand gestures. You can download it [here](https://20bn.com/datasets/jester).

After downloading all parts, extract using:

`cat 20bn-jester-v1-?? | tar zx`


### The network

We trained 3D Convolutional Neural Networks (3D-CNNs) on the dataset. 3D-CNNs take videos (set of images) as input and can directly extract spatio-temporal features for action recognition.

We aimed to use a model that had a good accuracy in classifying the gestures and at same time was not too computationally expensive. This is because the end goal of the system was to be able to control desktop applications using hand gestures and since most people don't have GPUs in their personal laptops, we wanted the model to be able to run robustly on CPUs.

We tried several 3D-CNN models to achieve this goal, some of which are present in the `lib` folder. 

After trying several models, we decided to use the **3D-CNN Super Lite model** (implemented in `lib/models.py` file) as it was able to achieve an accuracy of **86.15%** on the Jester test set and was able to run pleasantly on a Macbook Pro laptop.

### Gesture Control

After training our model on a GPU in Google Cloud, we downloaded the trained model's weights and architecture into the `models` directory. Then, we use `gesture-control/gesture.py` to capture the webcam video feed using OpenCV, load our pretrained model, perform inference and control our desired application by sending keystrokes based on the recognised gestures.


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


## Credits
1. https://20bn.com/datasets/jester
2. https://github.com/saetlan/20BN-jester
3. https://github.com/eleow/Gesture-Recognition-and-Control
4. https://github.com/patrickjohncyh/ibm-waldo/

## License
Distributed under the MIT License.
