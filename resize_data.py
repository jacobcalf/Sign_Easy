'''
# Jacob Calfee, Peter Hart, Louis Wenner
# Cleveland State University
# Senior Design Capstone August 2020-May 2021 - SignEasy: Dynamic American Sign Language Recognition with Machine Learning

 # File is responsible for collecting and writing OpenPose 2D skeleton coordinates. Program writes the 21 X,Y coordinates
    of the right-hand using OpenPose's hand detector. This program also resizes the data, so instead of each row representing
    one frame, we want one row to append all frames onto it. The number of columns will be 840 after appending.
 # Unique gestures were obtained with the following labels:
    0 --> Hello
    1 --> Thank You
    2 --> A (static)
    3 --> W (static)

 # NOTE: Pass '--number_people_max 1' as an argument for best efficiency
'''

import numpy
import csv


def resize():
    # Declare the variables
    n_steps = 20
    time_periods = 20
    number_sensors = 42
    label_column = number_sensors + 1

    # Load CSV data set with X (training data) and Y (label)
    raw_dataset = numpy.loadtxt("asl_signs.csv", delimiter=",")

    # Get the first 42 numbers on the line (the coordinates)
    X = raw_dataset[:, 0:number_sensors]

    # Get the frame stamp for each X,Y coordinate set
    #   Unused, but left for reference.
    time_stamps = raw_dataset[:, number_sensors]

    # Get the labels
    Y = raw_dataset[:, label_column]
    lap_buffer = []
    X, Y = create_segments_and_labels(X, time_periods, n_steps, Y)

    # Combine frames into one row
    for i in range(len(X) - 1):
        row = X[i] + [Y[i]]
        if 0.0 not in row[0:number_sensors]:
            # print(row)
            lap_buffer.append(row)

    # Make new .csv file that contains the frames in one row
    with open('asl_signs_one_row.csv', mode='w', newline='') as csv_file:
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

    # print("Segment: ", segments, "\n")

    return segments, label_array
