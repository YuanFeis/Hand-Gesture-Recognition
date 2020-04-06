#!/usr/bin/env python
# coding: utf-8

import pandas as pd

# read train csv into a pandas dataframe
training_labels_df = pd.read_csv('jester-v1-train.csv' , header=None,sep = ";", index_col=False)
# read validation csv into a pandas dataframe
validation_labels_df = pd.read_csv('jester-v1-validation.csv' , header=None,sep = ";", index_col=False)
# read the labels csv into a pandas dataframe
labels_df = pd.read_csv('/Users/samygarg/Downloads/jester-v1-labels.csv' , header=None,sep = ";", index_col=False)

print("Train size: " + str(training_labels_df.size))
print("Validation size: " + str(validation_labels_df.size))
print("Labels size: " + str(labels_df.size))

# names of labels to include in our filtered labels
targets_name = [
    "Thumb Down",
    "Stop Sign",
    "Swiping Left",
    "Swiping Right",
    "No gesture",
    "Thumb Up",
    "Turning Hand Clockwise",
    "Turning Hand Counterclockwise",
    "Shaking Hand",
    ]

# only keep labels that are present in the targets_name list
training_labels_filtered = training_labels_df[training_labels_df[1].isin(targets_name)]
validation_labels_filtered = validation_labels_df[validation_labels_df[1].isin(targets_name)]
labels_filtered = labels_df[labels_df[0].isin(targets_name)]

print("Train Filtered size: " + str(training_labels_filtered.size))
print("Validation Filtered size: " + str(validation_labels_filtered.size))
print("Labels Filtered size: " + str(labels_filtered.size))

# save our filtered dataframes as csv files
training_labels_filtered.to_csv('jester-v1-train-filtered.csv',header=False,sep=";",index=False)
validation_labels_filtered.to_csv('jester-v1-validation-filtered.csv',header=False,sep=";",index=False)
labels_filtered.to_csv('jester-v1-labels-filtered.csv',header=False,sep=";",index=False)