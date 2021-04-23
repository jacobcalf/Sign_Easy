'''
# Jacob Calfee, Peter Hart, Louis Wenner
# Cleveland State University
# Senior Design Capstone August 2020-May 2021 - SignEasy: Dynamic American Sign Language Recognition with Machine Learning

 # File is responsible for taking the output file from hand_data_collect and training it. Generally you don't want your
    to train its accuracy to 100% (or 1.0). Around when the data begins to level off is when the epoch should stop.
    Mess around with number of epochs to get just the right value!
 # Unique gestures were obtained with the following labels:
    0 --> Hello
    1 --> Thank You
    2 --> A (static)
    3 --> W (static)

 # NOTE: Pass '--number_people_max 1' as an argument for best efficiency
'''


import numpy as np
import matplotlib.pyplot as plt
import time
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.models import Sequential
from sklearn.model_selection import KFold
from sklearn.preprocessing import Normalizer
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


def main():
    # TODO CHANGE AS MORE LABELS ARE ADDED
    labels = ["Hello", "Thank You", "A", "W"]

    # The output dimension will be the same as the number of labels
    output_dimension = len(labels)

    # Parameters
    batch_size = 32
    num_epochs = 30
    seed = 7
    np.random.seed(seed)
    n_steps = 20
    time_periods = 20
    number_sensors = 42
    input_shape = (time_periods * number_sensors)

    # Load CSV dataset with X (training data) and Y (label)
    raw_dataset = np.loadtxt("asl_signs_one_row.csv", delimiter=",")  # right_hand_dataset_reduced10.csv

    # Get the first 840 numbers on the line (the coordinates)
    #   Each frame had 42 frames
    #   There is a total of 20 frames per data point
    #   There are 840 coordinate points per data point
    X = raw_dataset[:, 0:input_shape]

    # Get the frame stamp for each X,Y coordinate set
    time_stamps = raw_dataset[:, input_shape]

    # Get the labels
    Y = raw_dataset[:, input_shape]

    transformer = Normalizer().fit(X)

    X = transformer.transform(X)

    # Define the K-fold Cross Validation
    kfold = KFold(n_splits=10, shuffle=True)

    # Prepare to record data about each fold
    acc_per_fold = []
    loss_per_fold = []
    recall_per_fold = []
    precision_per_fold = []
    f1_per_fold = []

    fold_no = 1

    for train, test in kfold.split(X, Y):
        # Define Model
        model = Sequential()

        # First Layer
        model.add(Reshape((time_periods, number_sensors), input_shape=(input_shape,)))
        # Hidden Layers
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        # Final Layer
        model.add(Dense(output_dim=output_dimension, activation='softmax'))

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

        print("\n-----------------------------------------------------------")
        print(f"\nTraining for Fold #{fold_no}:")

        model.fit(X[train], Y[train], epochs=num_epochs, batch_size=batch_size, verbose=2, validation_data=(X[test], Y[test]), shuffle=True)

        # Evaluate Model
        results = model.evaluate(X[train], Y[train])

        y_pred = model.predict_classes(X[test])
        # y_pred = (predictions > 0.5)

        # Create confusion matrix
        cm = confusion_matrix(y_true=Y[test], y_pred=y_pred)

        # Prepare the presentation for the confusion matrix
        title = "Confusion Matrix for Gesture Classification"
        plt_confusion_matrix(cm, labels, title, fold_no)

        # Record the date the model was made
        t = time.localtime()
        d = time.strftime("date_%m-%d-%Y_time_%H-%M-%S", t)

        # Save model with timestamp and name into models folder
        model.save("Models/dynamic_kfold"
                   "-" + str(fold_no) + "_model_" + str(d) + ".h5")

        # Record various attributes about the model
        scores = model.evaluate(X[test], Y[test], verbose=0)
        acc = scores[1] * 100
        loss = scores[0]
        precision = precision_score(Y[test], y_pred, average="macro")
        recall = recall_score(Y[test], y_pred, average="macro")
        f1 = f1_score(y_true=Y[test], y_pred=y_pred, average='macro') * 100

        # Record the data for each fold:
        acc_per_fold.append(acc)
        loss_per_fold.append(loss)
        recall_per_fold.append(recall)
        precision_per_fold.append(precision)
        f1_per_fold.append(f1)

        # Print the results
        print(f"\nScores for fold #{fold_no}:")
        print(f"\tAccuracy: \t\t\t{acc}%")
        print(f"\tLoss: \t\t\t\t{loss}")
        print(f"\tPrecision score: \t{precision}")
        print(f"\tRecall score: \t\t{recall}")
        print(f"\tMacroF1 score: \t\t{f1}%")

        # We are done with this fold, so we increment to the next fold
        fold_no += 1

    print("\n\n\n-----------------------------------------------------------")
    print("-----------------------------------------------------------")
    print("\n\n\nOverall results from the folds:")

    m = sum(acc_per_fold) / len(acc_per_fold)
    print(f"\n\tAverage Accuracy is: \t\t\t{m}%")

    m = sum(loss_per_fold) / len(loss_per_fold)
    print(f"\n\tAverage Loss is: \t\t\t\t{m}")

    m = sum(precision_per_fold) / len(precision_per_fold)
    print(f"\n\tAverage Precision Score is: \t{m}")

    m = sum(recall_per_fold) / len(recall_per_fold)
    print(f"\n\tAverage Recall Score is: \t\t{m}")

    m = sum(f1_per_fold) / len(f1_per_fold)
    print(f"\n\tAverage MacroF1 score is: \t\t{m}%")

    print("\n\n\n-----------------------------------------------------------")
    print("-----------------------------------------------------------")

    # Show the confusion matrices
    # plt.show()


def plt_confusion_matrix(cm, labels, title, fold_no):
    # Makes everything blue, which is my favorite color
    cmap = plt.cm.Blues

    # Define the plot
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # Define the labels on the matrix display
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels,
           yticklabels=labels,
           title=title,
           ylabel='True',
           xlabel='Predicted')

    # Add values of confusion matrix to the plot
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    file_name = "Confusion_Matrices/Confusion-Matrix-" + str(fold_no) + ".png"
    plt.savefig(file_name)
    plt.close()


def create_segments_and_labels(df, time_steps, step, label_name):
    N_FEATURES = 42
    segments = np.empty((840))
    label_array = []
    np.set_printoptions(threshold=np.inf)
    for i in range(0, len(df) - time_steps, time_steps):
        base = df[i]

        for j in range(1, time_steps):
            base = np.append(base, df[i + j])

        segments = np.vstack((segments, base))
        label_array.append(label_name[i])

    print("Segment: ", segments.shape, "\n")
    label_array = np.asarray(label_array)
    segments = np.asarray(segments, dtype=np.float64)

    return segments, label_array


if __name__ == "__main__":
    main()
