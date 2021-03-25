import time
import numpy
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import argparse
import csv
import os
import sys
import cv2

def main():
    # TODO CHANGE OUTPUT_DIM WITH EVERY NEW SYMBOL
    output_dimension = 2

    # Hyper Parameters
    BATCH_SIZE = 32
    EPOCH = 24
    seed = 7
    numpy.random.seed(seed)
    n_steps = 20
    TIME_PERIODS = 20
    number_sensors = 42
    input_shape = (TIME_PERIODS * number_sensors)
    print(input_shape)
    print(TIME_PERIODS, " ", number_sensors)

    # Load CSV dataset with X (training data) and Y (label)
    raw_dataset = numpy.loadtxt("asl_signs.csv", delimiter=",")  # right_hand_dataset_reduced10.csv
    # Get the first 42 numbers on the line (the coordinates)
    X = raw_dataset[:, 0:42]

    # Get the frame stamp for each X,Y coordinate set
    time_stamps = raw_dataset[:, 42]

    # Get the labels
    Y = raw_dataset[:, 43]
    lap_buffer = []
    X, Y = create_segments_and_labels(X, TIME_PERIODS, n_steps, Y)

    for i in range(len(X) - 1):
        row = X[i] + [Y[i]]
        print(row)
        lap_buffer.append(row)

    with open('asl_signs_one_row.csv', mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # After break write data to row and start new lap
        for x in lap_buffer:
            writer.writerow(x)
        # Counting data for dynamic collection

def create_segments_and_labels(df, time_steps, step, label_name):
    N_FEATURES = 42
    segments = []
    label_array = []
    df = numpy.ndarray.tolist(df)
    for i in range(0, len(df) - time_steps, time_steps):
        base = df[i]

        for j in range(1, time_steps):
            base = base + df[i + j]

        segments.append(base)
        label_array.append(label_name[i])
        #print(segments.shape)

    print("Segment: ", segments, "\n")
    # reshaped = numpy.asarray(segments, dtype= numpy.float32).reshape(-1, time_steps, N_FEATURES)

    return segments, label_array

if __name__ == "__main__":
    main()
