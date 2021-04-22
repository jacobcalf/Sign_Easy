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

import resize_data
import argparse
import csv
import os
import sys
import cv2

dir_path = 'C:\\Users\\SignEasy\\Documents\\openpose-1.5.0'
sys.path.append("{}\\build\\python\\openpose\\Release".format(dir_path))
os.environ["PATH"] = os.environ["PATH"] + ";{}\\build\\x64\\Release;{}\\build\\bin;{};".format(dir_path, dir_path, dir_path)

try:
    import pyopenpose as op
except ImportError as e:
    print(e, file=sys.stderr)

print("Enter your desired laps:")
maxlapnum = int(input()) # Max lap is 1 indexed

print("Enter your desired rest time (in whole seconds):")
resttimems = int(input()) * 10 # in ms (10ms = 1s)

print("Enter what gesture you plan to collect:")
print("0 --> Hello\n1 --> Thank You\n2 --> A\n3 --> W")
gesture = int(input())

# Number of recording frames
# This should stay constant for all data recorded
maxrecordingframes = 20
rest_counter = ""

# openpose Flags
parser = argparse.ArgumentParser()
parser.add_argument("--hand", default=True)
parser.add_argument("--display", default=0)
parser.add_argument("--number_people_max", default=1)
args = parser.parse_known_args()

# openpose Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "{}\\models".format(dir_path)
params["hand"] = True
params["display"] = 0
params["number_people_max"] = 1

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
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    datum = op.Datum()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use webcam
    num_data_points = 0

    # Counting data for dynamic collection
    lap = 0
    lap_buffer = [[0 for i in range(44)] for j in range(20)]
    frame_counter = maxrecordingframes  # Did this so it starts on a Rest

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Unable to open webcam.")
            break

        # Use webcam frames
        datum.cvInputData = frame
        window_name = "Hand Detector Window"
        opWrapper.emplaceAndPop([datum])
        op_frame = datum.cvOutputData
        x_coord_list = []
        y_coord_list = []
        conf_lvl_sum = 0

        # After X clips, end collection
        if lap == maxlapnum:
            print("Done")
            break

        # Lap / frame printing
        resttimemin = maxrecordingframes
        resttimemax = resttimemin + resttimems
        if not (resttimemin <= frame_counter < resttimemax + 1):
                print("Lap #: ", lap, " Frame #: ", frame_counter)

        # Gives Xs frame break between data collections
        if frame_counter >= maxrecordingframes:

            # Rest counter and display for both screen and output
            secondsrest = resttimems // 10
            for x in range(secondsrest, 0, -1):
                if frame_counter == (resttimemax - 10*x):
                    if x == secondsrest:
                        print("")

                    rest_counter = "Rest. Starting again in " + str(x)
                    print(rest_counter)

                    if x == 1:
                        print("")
            cv2.putText(op_frame, rest_counter, (10,50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=3)
            frame_counter += 1

            if frame_counter == resttimemax + 1:
                cv2.putText(op_frame, "New Lap Starting", (10,50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=3)

                # Write to csv file
                with open('asl_signs.csv', mode='a', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    # After break write data to row and start new lap
                    for x in lap_buffer:
                        writer.writerow(x)
                    # Counting data for dynamic collection

                lap += 1
                lap_buffer = [[0 for i in range(44)] for j in range(20)]
                frame_counter = 0
        else:

            try:
                # Enter coordinates for right hand
                rightHandKeypoints = datum.handKeypoints[1]
                for entries in rightHandKeypoints:
                    for coordinates in entries:
                        x_coord_list.append(coordinates[0])
                        y_coord_list.append(coordinates[1])
                        conf_lvl_sum += (coordinates[2])

                # Build Lists of X,Y + label
                conf_lvl_avg = (conf_lvl_sum/21)

                # Accept the data above conf_level -->  write, append label
                if conf_lvl_avg >= 0.15:
                    x_y_coord_list = x_coord_list + y_coord_list
                    row = x_y_coord_list + [frame_counter] + [gesture]
                    lap_buffer[frame_counter] = row

                    # Frame data has been recorded, onto next frame
                    frame_counter += 1

            # If data wasn't collected for a frame, restart the lap.
            except:
                frame_counter = 0
                lap_buffer = [[0 for i in range(44)] for j in range(20)]
                print("Restarting collection.")
                pass

            # Prints lab and frame counter in screen
            top_left = "Lap " + str(lap) + "/" + str(maxlapnum)
            bottom_left = "Frame " + str(frame_counter) + "/" + str(maxrecordingframes)
            cv2.putText(op_frame, top_left, (10,50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=3)
            cv2.putText(op_frame, bottom_left, (10,100), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=3)

        cv2.imshow(window_name, op_frame)
        # Hit q to terminate the openpose window, and exit program
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            print("Terminating Program")
            exit()

    # We are done recording, so we close the capture
    cap.release()

except Exception as e:
     print(e)
     sys.exit(-1)

print("\n\nCreating one row file....")

# Prepare data
resize_data.resize()

print("\n\n All done!")
