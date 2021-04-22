'''
# Jacob Calfee, Peter Hart, Louis Wenner
# Cleveland State University
# Senior Design Capstone August 2020- May 2021 - SignEasy: Dynamic American Sign Language Recognition with Machine Learning

 # Program functions as a demo by using the trained .h5 TensorFlow Keras model for Hand Gesture Recognition.
    By reading webcam frames, the user's right hand data is used as input to make a prediction on the corresponding hand
    gesture from the classes created.
 # The user performs the action during the 2 seconds aloted (20 frames)
 # The program displays the model's predicted hand symbol's string in the top left near "Prediction".
 # This project was built upon the hand gesture recognition project by Jacob Calfee and Michael Fasko
    Hackron 4k (10/5/19 - 10/6/19) - Hand Gesture Recognition with Machine Learning
    github: https://github.com/Fasko/Hand-Gesture-Recognition

    0 --> Hello
    1 --> Thank You
    2 --> A (static)
    3 --> W (static)

 # NOTE: Pass '--number_people_max 1' as an argument for best efficiency
'''

import argparse
import cv2
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import Normalizer
import os
import sys


def main():

    dir_path = 'C:\\Users\\SignEasy\\Documents\\openpose-1.5.0'
    sys.path.append("{}\\build\\python\\openpose\\Release".format(dir_path))
    os.environ["PATH"] = os.environ["PATH"] + ";{}\\build\\x64\\Release;{}\\build\\bin;{};".format(dir_path, dir_path, dir_path)

    try:
        import pyopenpose as op
    except ImportError as e:
        print(e, file=sys.stderr)

    # OpenPose Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", default=True)
    parser.add_argument("--display", default=0)
    parser.add_argument("--number_people_max", default=1)

    args = parser.parse_known_args()

    # OpenPose Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "{}\\models".format(dir_path)
    params["hand"] = True
    params["display"] = 0
    params["number_people_max"] = 1

    # Add others command line arguments
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1]) - 1:
            next_item = args[1][i + 1]
        else:
            next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-', '')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-', '')
            if key not in params: params[key] = next_item
    try:
        import tensorflow as tf
        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()
        datum = op.Datum()

        # Start TensorFlow, load Keras model
        config = tf.ConfigProto()

        config.gpu_options.per_process_gpu_memory_fraction = 0.1  # Only allocates a portion of VRAM to TensorFlow
        session = tf.Session(config=config)

        # TODO CHANGE WITH EVERY NEW MODEL
        tf_model = load_model('Models/dynamic_kfold-1_model_date_04-22-2021_time_16-20-09.h5') # 'normalized_epochs200_10_data_points10_06_2019_02_00_54.h5

        # Capture Frames
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use webcam
        num_data_points = 0
        frame_counter = 0
        result = ""
        xy_set = []
        conf_level_sum = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Use Webcam frames, render OpenPose
            datum.cvInputData = frame
            opWrapper.emplaceAndPop([datum])
            op_frame = datum.cvOutputData
            window_name = "Hand Classification Window"

            # All available hand keypoints (OpenPose 1.5 (0-20))
            hand_data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

            # TODO Add new gestures to this
            prediction_strings = ["Hello", "Thank You", "A", "W"]

            x_set =[]
            y_set = []

            num_predictions = len(prediction_strings)

            # Ensure hand keypoints exist before doing classification
            try:
                if 0 <= frame_counter < 20:
                    rightHandKeypoints = datum.handKeypoints[1]
                    for entries in rightHandKeypoints:
                        for hand_entry in hand_data:
                            conf_level_sum += entries[hand_entry][2]
                            x_set.append(entries[hand_entry][0])
                            y_set.append(entries[hand_entry][1])

                    xy_set = xy_set + x_set
                    xy_set = xy_set + y_set

                    bottom_left = "Frame: " + str(frame_counter)
                    cv2.putText(op_frame, bottom_left, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)

                    frame_counter += 1
                elif frame_counter == 20:
                    xy_set = np.asarray(xy_set, dtype=np.float32)
                    xy_set = xy_set.reshape(1, -1)
                    transformer = Normalizer().fit(xy_set)
                    X_test = transformer.transform(xy_set)

                    # Prediction occurs here
                    predictions = tf_model.predict(xy_set)
                    predictions = predictions.flatten()

                    # Issue here is that conf_level_sum works until last iteration
                    print("conf_level_sum: " + str(conf_level_sum))
                    conf_level = conf_level_sum / num_predictions / 10
                    print("conf_level: " + str(conf_level))

                    if conf_level > .85:
                        predictionToDisplay = prediction_strings[np.argmax(predictions)]
                    else:
                        predictionToDisplay = "N/A"

                    print("prediction: " + str(predictionToDisplay))

                    conf_level_sum = 0
                    frame_counter += 1

                elif 21 <= frame_counter < 50:
                    # Show prediction
                    cv2.putText(op_frame, "Result: "+predictionToDisplay, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0),
                                thickness=2)

                    frame_counter += 1

                elif frame_counter == 50:
                    frame_counter = 0
                    xy_set = []
                    # clear prediction

            except Exception as e:
                cv2.putText(op_frame, "Restarting, please put hand in frame.", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0),
                            thickness=2)
                print(e)
                conf_level_sum = 0
                frame_counter = 0
                xy_set = []
                pass

            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.imshow(window_name, op_frame)

            if cv2.waitKey(1) == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                print("Terminating Program")
                exit()

    except Exception as e:
         print(e)
         sys.exit(-1)


if __name__ == "__main__":
    main()
