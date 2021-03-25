'''
# Jacob Calfee, Peter Hart, Louis Wenner
# Cleveland State University
# Senior Design Capstone August 2020-May 2021 - SignEasy: Dynamic American Sign Language Recognition with Machine Learning

 # File is responsible for taking the output file from hand_data_collect and training it. Generally you don't want your
    to train its accuracy to 100% (or 1.0). Around when the data begins to level off is when the epoch should stop.
    Mess around with number of epochs to get just the right value!
 # Unique gestures were obtained with the following labels:
    0 --> J
    1 --> Z
    2 --> A

 # NOTE: Pass '--number_people_max 1' as an argument for best efficiency
'''

import time
import numpy
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer


def main():
    # TODO CHANGE OUTPUT_DIM WITH EVERY NEW SYMBOL
    print("Enter number of gestures you are training:")
    output_dimension = int(input())

    print("Enter number of epochs:")
    EPOCH = int(input())

    # Parameters
    BATCH_SIZE = 32
    seed = 7
    numpy.random.seed(seed)
    n_steps = 20
    TIME_PERIODS = 20
    number_sensors = 42
    input_shape = (TIME_PERIODS * number_sensors)

    # Load CSV dataset with X (training data) and Y (label)
    raw_dataset = numpy.loadtxt("asl_signs_one_row.csv", delimiter=",")  # right_hand_dataset_reduced10.csv

    # Get the first 42 numbers on the line (the coordinates)
    X = raw_dataset[:, 0:840]

    # Get the frame stamp for each X,Y coordinate set
    time_stamps = raw_dataset[:, 840]

    # Get the labels
    Y = raw_dataset[:, 840]

    transformer = Normalizer().fit(X)

    X = transformer.transform(X)

    # Randomize Test/Train Splits
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=0)

    # Model
    model = Sequential()
    model.add(Reshape((TIME_PERIODS, number_sensors), input_shape=(input_shape,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(output_dim=output_dimension, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    model.fit(X_train, Y_train,
              epochs=EPOCH,
              batch_size=BATCH_SIZE,
              verbose=2,
              validation_data=(X_test, Y_test),
              shuffle=True)

    # Evaluate Model
    results = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)
    print('test loss, test acc:', results)

    named_tuple = time.localtime()
    time_string = time.strftime("%m_%d_%Y_%H_%M_%S", named_tuple)

    # Save model w/ timestamp and name
    model.save("test_dynamic_" + time_string + ".h5")


def create_segments_and_labels(df, time_steps, step, label_name):
    N_FEATURES = 42
    segments = numpy.empty((840))
    label_array = []
    numpy.set_printoptions(threshold=numpy.inf)
    for i in range(0, len(df) - time_steps, time_steps):
        base = df[i]

        for j in range(1, time_steps):
            base = numpy.append(base, df[i + j])

        segments = numpy.vstack((segments, base))
        label_array.append(label_name[i])

    print("Segment: ", segments.shape, "\n")
    label_array = numpy.asarray(label_array)
    segments = numpy.asarray(segments, dtype=numpy.float64)

    return segments, label_array


if __name__ == "__main__":
    main()
