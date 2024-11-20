import csv
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence = 0.3)

label_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

labels_dict = {}
for i, dir_ in enumerate(label_names):
    labels_dict[dir_] = i
#csv
file = open('handlandmarkss.csv','w',newline='')
writer = csv.writer(file)
writer.writerow(["label"] + ["x_{}".format(i) for i in range(21)] + ["y_{}".format(i) for i in range(21)]) #header row w/ class labels n coordinates for x y landmarks

cap = cv2.VideoCapture(0)

current_label = 0
max_samples = 20
#sample count for each label e.g [0,0,0,0] counts elements
samples_captured = [0] * len(label_names)

try:
    while True:
        ret, frame = cap.read()
        frame_rbg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rbg)

        cv2.putText(frame, "Current label: {}".format(label_names[current_label]),(10,30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
        cv2.putText(frame, "Samples captured: {}/{}".format(samples_captured[current_label], max_samples), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)

                #relative 
                landmark_list = [[hand_landmarks.landmark[i].x,hand_landmarks.landmark[i].y] for i in range(len(hand_landmarks.landmark))]
                #flattens list
                landmark_list = [x for pair in landmark_list for x in pair]

                #convert to relative coordinates, where origin is first landmark
                base_x, base_y = landmark_list[0],landmark_list[1]
                landmark_list = [(x - base_x) for x in landmark_list[::2]] + [(y - base_y) for y in landmark_list[1::2]]
                #maximum absolute value
                max_value = max(list(map(abs, landmark_list)))
                #normalise 
                landmark_list = [n / max_value for n in landmark_list]

                key = cv2.waitKey(1) & 0xFF

                #ord ascii value 
                if key == ord(' '):
                    writer.writerow([labels_dict[label_names[current_label]]]+landmark_list)
                    #console
                    print("Label {} logged".format(label_names[current_label]))
                    samples_captured[current_label] += 1
                    if samples_captured[current_label] >= max_samples:
                        current_label = (current_label + 1 ) % len(label_names)
                #move bk and forth between label
                elif key == ord('a'):
                    current_label = (current_label - 1) % len(label_names)
                elif key == ord('d'):
                    current_label = (current_label + 1) % len(label_names)
        cv2.imshow('frame',frame)
        #close 
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    cap.release()
    file.close()
    cv2.destroyAllWindows()
