# Hand-Gesture-Recognition

Automatic detection and classification of dynamic hand gestures in real-world environments and its application in human-computer interfaces is a challenging task that has gained significant scientific interest in recent years. 

This project aims to implement an end-to-end system that performs simultaneous detection and classification of dynamic hand gestures from multimodal data and further, uses the detected gestures to control desktop applications. The dataset used is the [Jester](https://20bn.com/datasets/jester) dataset which contains 148K short clips of videos that depict a person performing a gesture in front of a camera.


## 1. How it works?

### 1.1 The dataset

We used the 20BN-JESTER dataset, which is a large collection of densely-labeled video clips that show humans performing pre-defined hand gestures in front of a laptop camera or webcam. It consists of 148,092 labeled videos, depicting 25 different classes of human hand gestures and allows for training robust machine learning models to recognize human hand gestures. You can download it [here](https://20bn.com/datasets/jester).


### 1.2 The network

We trained 3D Convolutional Neural Networks (3D-CNNs) on the dataset. 3D-CNNs take videos (set of images) as input and can directly extract spatio-temporal features for action recognition.

We aimed to use a model that had a good accuracy in classifying the gestures and at same time was not too computationally expensive. This is because the end goal of the system was to be able to control desktop applications using hand gestures and since most people don't have GPUs in their personal laptops, we wanted the model to be able to run robustly on CPUs.

We tried several 3D-CNN models to achieve this goal, some of which are present in the `lib` folder. 

After trying several models, we decided to use the **3D-CNN Super Lite model** (implemented in `lib/models.py` file) as it was able to achieve an accuracy of **86.15%** on the Jester test set and was able to run pleasantly on a Macbook Pro laptop.

### 1.3 Gesture Control

After training our model on a GPU in Google Cloud, we downloaded the trained model's weights and architecture into the `models` directory. Then, we use `gesture-control/gesture.py` to capture the webcam video feed using OpenCV, load our pretrained model, perform inference and control our desired application by sending keystrokes based on the recognised gestures.


## 2. Instructions

### 2.1 Using Pretrained Models (Recommended)

#### 2.1.1 Set up the system
1. Fork this repository and clone the forked repository.
2. Change directory using `cd Hand-Gesture-Recognition`.
3. Create a Python 3.7 virtual environment using conda/pipenv/virtualenv and activate it.
4. Use `pip install -r requirements.txt` to install the requirements.

#### 2.1.2 Modify the config file
In the `gesture-control` folder, you will find the `gesture_control_config.cfg` config file. This file contains the configuration for running `gesture.py`. You can run it with the default parameters or modify it, if using a custom model.

#### 2.1.3 Run the gesture control demo
1. Change to the `gesture-control` directory using `cd gesture-control`.
2. Run `python hanuman.py --config "gesture_control_config.cfg"`.
3. Wait for the program to load. Open a YouTube video in your browser and click anywhere on the page to focus.
4. Now, perform any of the gestures specified in the mapping.ini file to control YouTube.


### 2.2 Using Your Own Model

#### 2.2.1 Set up the system
Follow the same steps as in 2.1.1.

#### 2.2.2 Download and Extract the Jester Dataset
In order to train your own model, you will need to download TwentyBN's Jester Dataset. This dataset has been made available under the Creative Commons Attribution 4.0 International license CC BY-NC-ND 4.0 and can be used for academic research free of charge. In order to get access to the dataset, you will need to register on their website.

Jester dataset has been provided as one large TGZ archive and with a total download size of 22.8 GB, split into 23 parts of about 1 GB each. After downloading all the parts, you can extract the videos using:

`cat 20bn-jester-v1-?? | tar zx`

#### 2.2.3 Modify the config file
In the root folder, you will find `config.cfg`. This file needs to be modified to indicate the location of both the CSV files and the videos from the Jester dataset as well the parameters to be used during training and testing.

#### 2.2.4 Create your own model
`lib` folder already has 3 3D-CNN models and a RESNET3D model implemented for you to use. You can modify the model.py file to create your own 3D architecture.

#### 2.2.5 Train your model
After configuring your `config.cfg`, and creating your model, you can train the model using `train.py`.

Due to the very large size of the Jester dataset, it is strongly recommended that you only perform the training using a GPU.

You can train the model using - `python train.py --config "config.cfg"`

#### 2.2.6 Download your trained model 
After training your model, move the model json and weights file to the `models` folder.

#### 2.2.7 Test your trained model on Jester test set
Update `path_weights` in `config.cfg` with the path of your model's weights and run `python test.py --config "config.cfg"` to test your model on the Jester test set. This will generate a `prediction.csv` file with the predicted labels for the test set.

#### 2.2.8 Run the gesture control demo
Follow 2.1.2 and 2.1.3 to control applications using your your trained model.


## 3. Credits
1. https://20bn.com/datasets/jester
2. https://github.com/saetlan/20BN-jester
3. https://github.com/eleow/Gesture-Recognition-and-Control
4. https://github.com/patrickjohncyh/ibm-waldo/

## 4. License
Distributed under the MIT License.
