import  cv2
import os
import sys
import csv
import numpy as np
from collections import deque
import pyautogui
import argparse
import configparser
from ast import literal_eval
import errno
import tensorflow as tf

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from lib.data_loader import FrameQueue


# function to load the pretrained CNN model
def load_model(model_json_path, model_weights_path):
    # read the model json file
    json_file = open(model_json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    # load the model from json file
    model = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(model_weights_path)

    print("Loaded CNN model from disk")

    return model


def main(args):
    # extract information from the configuration file
    nb_frames                = config.getint('general', 'nb_frames')
    target_size              = literal_eval(config.get('general', 'target_size'))
    nb_classes               = config.getint('general', 'nb_classes')
    csv_labels               = config.get('path', 'csv_labels')
    gesture_keyboard_mapping = config.get('path', 'gesture_keyboard_mapping')
    model_json_path          = config.get('path', 'model_json_path')
    model_weights_path       = config.get('path', 'model_weights_path')


    # activate camera on laptop
    cap = cv2.VideoCapture(0)

    # set a lower resolution for speed up
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)


    # open our labels csv file
    with open(csv_labels)as f:
        # read it using the csv reader
        f_csv = csv.reader(f)
        # create empty list to store the labels list
        labels_list = []
        # for each row in the csv file
        for row in f_csv:
            # append the label to the labels list
            labels_list.append(row)
        # convert labels list to a tuple
        labels_list = tuple(labels_list)


    # load the gesture -> key mapping from config
    mapping = configparser.ConfigParser()
    action = {}
    # if the specified mapping file exists
    if os.path.isfile(gesture_keyboard_mapping):
        # read the file
        mapping.read(gesture_keyboard_mapping)
        # for each mapping in the mapping file
        for m in mapping['MAPPING']: 
            # get the value
            val = mapping['MAPPING'][m].split(',')
            # set the action
            action[m] = {'fn': val[0], 'keys': val[1:]}  # fn: hotkey, press, typewrite
    # if the mapping file is not present
    else:
        # raise error
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), args.mapping)

    # create a deque to store the identified gestures
    act = deque(['No gesture', "No gesture"], maxlen=3)

    # load our pretrained model
    net = load_model(model_json_path, model_weights_path)

    # initialize our frame queue
    frame_queue = FrameQueue(nb_frames, target_size)

    # while the video is running
    while(cap.isOpened()):
        # read frame
        on, frame = cap.read()

        # if the video is on
        if on == True:
            # calibrate the channels
            b, g, r = cv2.split(frame)
            frame_calibrated = cv2.merge([r, g, b])

            # generate the batch for model
            batch_x = frame_queue.img_in_queue(frame_calibrated)

            # get predictions for the batch
            res = net.predict(batch_x)

            # get the predicted class
            predicted_class = labels_list[np.argmax(res)]
            print('Predicted Class = ', predicted_class, 'Accuracy = ', np.amax(res)*100,'%')

            # if the max probability of result is greater than threshold then 
            # set the gesture to the predicted label
            # else set to "No Gesture"
            gesture = (labels_list[np.argmax(res)] if max(res[0]) > 0.8 else labels_list[8])[0]
            # print(gesture)

            # convert the gesture to lowercase
            gesture = gesture.lower()
            # append the gesture to the act queue
            act.append(gesture)

            # if the first gesture in act queue is not the same as the second one
            # and the number of unique elements in act queue is 1
            if (act[0] != act[1] and len(set(list(act)[1:])) == 1):
                # print(action.keys())
                # if gesture is in the mapping
                if gesture in action.keys():       
                    t = action[gesture]['fn']
                    k = action[gesture]['keys']

                    print('[DEBUG]', gesture, '-- ', t, str(k))

                    if t == 'typewrite':
                        pyautogui.typewrite(k)
                    elif t == 'press':
                        pyautogui.press(k)
                    elif t == 'hotkey':
                        for key in k:
                            pyautogui.keyDown(key)
                        for key in k[::-1]:
                            pyautogui.keyUp(key)


            cv2.imshow('camera0', frame)

            # exit when pressing key q
            if(cv2.waitKey(1)&0xFF) == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config", help="Configuration file to run the script", required=True)
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    main(config)







