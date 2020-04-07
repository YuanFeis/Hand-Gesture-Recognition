import pandas as pd
import numpy as np
import lib.image as kmg


class DataLoader():
    """ Class used to load csvs
    # Arguments
        path_vid    : path to the folder containing the extracted Jester dataset
        path_labels : path to the csv containing the labels
        path_train  : path to the csv containing the list of videos for training
        path_val    : path to the csv containing the list of videos used for validation
        path_test   : path to the csv containing the list of videos used for test
    # Returns
        An instance of the DataLoader class  
    """
    def __init__(self, path_vid, path_labels, path_train=None, path_val=None, path_test=None):
        self.path_vid    = path_vid
        self.path_labels = path_labels
        self.path_train  = path_train
        self.path_val    = path_val
        self.path_test   = path_test

        self.get_labels(path_labels)

        if self.path_train:
            self.train_df = self.load_video_labels(self.path_train)

        if self.path_val:
            self.val_df = self.load_video_labels(self.path_val)

        if self.path_test:
            self.test_df = self.load_video_labels(self.path_test, mode="input")

    def get_labels(self, path_labels):
        """Loads the labels dataframe from a csv and creates dictionaries to convert the string labels to int and backwards
        # Arguments
            path_labels : path to the csv containing the labels
        """
        # read the labels csv into a dataframe using pandas
        self.labels_df = pd.read_csv(path_labels, names=['label'])
        # extract the list of labels from the dataframe
        self.labels = [str(label[0]) for label in self.labels_df.values]
        # get the length of the list of labels
        self.no_of_labels = len(self.labels)
        # create a dictionary mapping labels to integers
        self.label_to_int = dict(zip(self.labels, range(self.no_of_labels)))
        # create a dictionary mapping integers to labels
        self.int_to_label = dict(enumerate(self.labels))

    def load_video_labels(self, path_subset, mode="label"):
        """ Loads a Dataframe from a csv
        # Arguments
            path_subset : String, path to the csv to load
            mode        : String, (default: label), if mode is set to "label", filters rows if the labels exists in the labels Dataframe loaded previously
        # Returns
            A DataFrame
        """
        if mode=="input":
            names=['video_id']
        elif mode=="label":
            names=['video_id', 'label']
        
        df = pd.read_csv(path_subset, sep=';', names=names) 
        
        if mode == "label":
            df = df[df.label.isin(self.labels)]

        return df
    
    def categorical_to_label(self, vector):
        """ Convert a vector to its associated string label
        # Arguments
            vector : Vector representing the label of a video
        # Returns
            Returns a String that is the label of a video
        """
        return self.int_to_label[np.where(vector==1)[0][0]]


class FrameQueue(object):
    """ Class used to create a queue from video frames
    # Arguments
        nb_frames   : no of frames for each video
        target_size : size of each frame
    # Returns
        A batch of frames as input for the model  
    """
    def __init__(self, nb_frames, target_size):
        self.target_size = target_size
        self.nb_frames = nb_frames

        # create a new array of given shape, filled with zeros
        # representing the batch of frames
        self.batch = np.zeros((1, self.nb_frames) + target_size + (3,))

    def img_in_queue(self, img):
        # for i in (0, no_of_frames - 1) i.e. (0, 15)
        for i in range(self.batch.shape[1] - 1):
            self.batch[0, i] = self.batch[0, i+1]
        # load image from array and resize it
        img = kmg.load_img_from_array(img, target_size=self.target_size)
        # convert the resized image to numpy array
        x = kmg.img_to_array(img)
        # normalize the image and add to batch
        self.batch[0, self.batch.shape[1] - 1] = x / 255

        return self.batch
