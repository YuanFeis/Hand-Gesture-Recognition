import argparse
import configparser
from ast import literal_eval

import os
from math import ceil
import numpy as np

import lib.image as kmg
from lib.data_loader import DataLoader
from lib.utils import mkdirs
import lib.model as model
from lib.model_res import Resnet3DBuilder

from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD

def main(args):
    # extract information from the configuration file
    nb_frames    = config.getint('general', 'nb_frames')
    skip         = config.getint('general', 'skip')
    target_size  = literal_eval(config.get('general', 'target_size'))
    batch_size   = config.getint('general', 'batch_size')
    epochs       = config.getint('general', 'epochs')
    nb_classes   = config.getint('general', 'nb_classes')

    model_name   = config.get('path', 'model_name')
    data_root    = config.get('path', 'data_root')
    data_model   = config.get('path', 'data_model')
    data_vid     = config.get('path', 'data_vid')

    path_weights = config.get('path', 'path_weights')

    csv_labels   = config.get('path', 'csv_labels')
    csv_train    = config.get('path', 'csv_train')
    csv_val      = config.get('path', 'csv_val')
    csv_test     = config.get('path', 'csv_test')

    workers              = config.getint('option', 'workers')
    use_multiprocessing  = config.getboolean('option', 'use_multiprocessing')
    max_queue_size       = config.getint('option', 'max_queue_size')

    # join together the needed paths
    path_vid = os.path.join(data_root, data_vid)
    path_model = os.path.join(data_root, data_model, model_name)
    path_labels = os.path.join(data_root, csv_labels)
    path_train = os.path.join(data_root, csv_train)
    path_val = os.path.join(data_root, csv_val)
    path_test = os.path.join(data_root, csv_test)

    # Input shape of the input Tensor
    inp_shape   = (nb_frames,) + target_size + (3,)

    # load the data using DataLoader class
    data = DataLoader(path_vid, path_labels, path_train, path_val)

    # create model folder
    mkdirs(path_model, 0o755)

    # create the generators for the training and validation set
    gen = kmg.ImageDataGenerator()
    gen_train = gen.flow_video_from_dataframe(data.train_df, path_vid, path_classes=path_labels, x_col='video_id', y_col="label", target_size=target_size, batch_size=batch_size, nb_frames=nb_frames, skip=skip, has_ext=True)
    gen_val = gen.flow_video_from_dataframe(data.val_df, path_vid, path_classes=path_labels, x_col='video_id', y_col="label", target_size=target_size, batch_size=batch_size, nb_frames=nb_frames, skip=skip, has_ext=True)
    
    # MODEL

    # # Build and compile RESNET3D model
    # net = Resnet3DBuilder.build_resnet_101(inp_shape, nb_classes, drop_rate=0.5)
    # opti = SGD(lr=0.01, momentum=0.9, decay= 0.0001, nesterov=False)
    # net.compile(optimizer=opti,
    #             loss="categorical_crossentropy",
    #             metrics=["accuracy"]) 

    # Build and compile CNN3D Lite model
    net = model.CNN3D_lite(inp_shape=inp_shape, nb_classes=nb_classes)
    net.compile(optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy", "top_k_categorical_accuracy"]) 

    # if model weights file is present
    # load the model weights
    if(path_weights != "None"):
        print("Loading weights from : " + path_weights)
        net.load_weights(path_weights)

    # file format for saving the best model
    model_file_format_best = os.path.join(path_model,'model.best.hdf5') 

    # checkpoint the best model
    checkpointer_best = ModelCheckpoint(model_file_format_best, monitor='val_accuracy',verbose=1, save_best_only=True, mode='max')

    # get the number of samples in the training and validation set
    nb_sample_train = data.train_df["video_id"].size
    nb_sample_val   = data.val_df["video_id"].size

    # launch the training 
    net.fit_generator(
            generator=gen_train,
            steps_per_epoch=ceil(nb_sample_train/batch_size),
            epochs=epochs,
            validation_data=gen_val,
            validation_steps=ceil(nb_sample_val/batch_size),
            shuffle=True,
            verbose=1,
            workers=workers,
            max_queue_size=max_queue_size,
            use_multiprocessing=use_multiprocessing,
            callbacks=[checkpointer_best],
    )

    # after training serialize the final model to JSON
    model_json = net.to_json()
    with open("radhakrishna_all.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    net.save_weights("radhakrishna_all.h5")
    print("Saved model to disk")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config", help="Configuration file to run the script", required=True)
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    main(config)
