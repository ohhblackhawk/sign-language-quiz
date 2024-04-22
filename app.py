# app.py

from flask import Flask, render_template, request, jsonify, Response, make_response, send_file,redirect, url_for
import cv2
import mediapipe as mp 
import re

import numpy as np
import pickle
import random
import time
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/signs')
def signs():
    return render_template('signs.html')

#caches images
@app.route('/static/images/<letter>.gif')
def send_image(letter):
    #cache timeout 1 hr
    cache_timeout = 3600
    #attaching headers
    response = make_response(send_file('static/images/{}.gif'.format(letter)))
    response.headers['Cache-Control'] = 'max-age={}, public'.format(cache_timeout)
    return response

#difficulty
@app.route('/difficulty')
def difficulty():
    return render_template('difficulty.html')

#choose difficulty
@app.route('/quiz/<difficulty>', methods=['GET'])
def quiz(difficulty):
    if difficulty == 'easy':
        word_pool = easy_word_pool
    elif difficulty == 'medium':
        word_pool = medium_word_pool
    elif difficulty == 'hard':
        word_pool = hard_word_pool
    elif difficulty == 'custom':
        word_pool = custom_word_pool
    else:
        return "Invalid Difficulty Level"
    return render_template('quiz.html', word_pool=word_pool, difficulty=difficulty)

#word pool
easy_word_pool = ['cat','dog','fish','hat']
medium_word_pool = ['sasha', 'many', 'minion', 'stuart']
hard_word_pool = ['banana', 'zebra', 'jux', 'sticky']
custom_word_pool = []

#for custom word
@app.route('/add_word', methods=['POST'])
def add_word():
    customwords = request.form['customwords']
    words = []
    #validation
    for word in customwords.split(','):
        word = word.strip().lower()
        if re.match('^[a-z]+$',word):
            words.append(word)
        else:
            return render_template('difficulty.html', error_message="Invalid Entry!")
    for word in words:
        custom_word_pool.append(word)
    return redirect(url_for('quiz',difficulty='custom'))


# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Define the alphabet dictionary
alphabet_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

word_list = ["abca","abba"]
word_to_spell = random.choice(word_list)
word_spelled = ""


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_word', methods=['GET'])
def update_word():
    global word_to_spell, letter
    word_to_spell = random.choice(word_list)
    letter = word_to_spell[0]  # Get the first letter of the word
    return jsonify({'word_to_spell': word_to_spell, 'letter': letter})


@app.route('/get_word_list', methods=['GET'])
def get_word_list():
    global word_list
    return jsonify({'word_list': word_list})

@app.route('/get_current_word', methods=['GET'])
def get_current_word():
    global word_to_spell
    return jsonify({'word': word_to_spell})

guessed_letters = []  # Define the guessed_letters list as a global variable


start_time = time.time()  # Initialize the start time
time_limit = 60  # Set the time limit in seconds

@app.route('/get_time_remaining', methods=['GET'])
def get_time_remaining():
    global start_time, time_limit
    current_time = time.time()
    time_elapsed = current_time - start_time
    time_remaining = max(0, time_limit - time_elapsed)
    return jsonify({'time_remaining': time_remaining})

last_update_time = 0  # Initialize a variable to store the last update time


@app.route('/update_letter', methods=['GET'])
def update_letter():
    global word_to_spell, letter, guessed_letters
    if len(word_to_spell) > 0:
        if letter in guessed_letters:
            pass  # do nothing if the letter has already been guessed
        else:
            guessed_letters.append(letter)  # Add the guessed letter to the list
            if letter == word_to_spell[0]:  # check if the guessed letter is correct
                word_to_spell = word_to_spell[1:]  # Remove the first letter from the word
                if len(word_to_spell) > 0:
                    letter = word_to_spell[0]  # Get the new first letter
                else:
                    letter = ""  # Reset the current letter if the word is complete
            else:
                pass  # do nothing if the guessed letter is incorrect
    return jsonify({'letter': letter})


@app.route('/update_message', methods=['GET'])
def update_message():
    global word_spelled, word_to_spell  # <--- Add word_to_spell here
    message = ''
    if word_spelled == word_to_spell:
        if time.time() - start_time > 5:  # wait 5 seconds before displaying "already spelled" message
            message = "You've already spelled the word correctly Congratulations!"
            word_to_spell = random.choice(word_list)
            word_spelled = ""
        else:
            message = "Please wait..."
    else:
        message = "Try again!"
    return jsonify({'message': message})

def gen_frames():
    global word_spelled, word_to_spell, start_time
    camera = cv2.VideoCapture(0)
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                image = cv2.flip(image, 1)
                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        landmarks = np.array([[landmark.x, landmark.y] for landmark in hand_landmarks.landmark])
                        normalized_landmarks = normalize_landmarks(landmarks)
                        letter_index = model.predict([normalized_landmarks])[0]
                        letter = alphabet_dict[int(letter_index)]

                        cv2.putText(image, letter, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                        if letter in word_to_spell:
                            word_spelled += letter
                            if letter == word_to_spell[0]:
                                word_to_spell = word_to_spell[1:]
                            else:
                                word_spelled = word_spelled[:-1]  # remove the incorrect letter
                        else:
                            word_spelled = ""  # reset the word_spelled variable

                        if word_spelled == word_to_spell:
                            start_time = time.time()  # record the time when the word is spelled correctly

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
def normalize_landmarks(landmarks):
    x_min, x_max = np.min(landmarks[:, 0]), np.max(landmarks[:, 0])
    y_min, y_max = np.min(landmarks[:, 1]), np.max(landmarks[:, 1])

    normalized_landmarks = (landmarks - np.array([x_min, y_min])) / np.array([x_max - x_min, y_max - y_min])
    return normalized_landmarks.flatten()


# def gen_frames():
#     camera = cv2.VideoCapture(0)
#     mp_drawing = mp.solutions.drawing_utils
#     mp_hands = mp.solutions.hands
#     with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
#         while True:
#             success, frame = camera.read()
#             if not success:
#                 break
#             else:
#                 image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#                 #flips video so right hand displays on the right, and output is a mirror for user. 
#                 image = cv2.flip(image,1)
#                 image.flags.writeable = False
#                 results = hands.process(image)
#                 image.flags.writeable = True
#                 image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#                 if results.multi_hand_landmarks:
#                     for hand_landmarks in results.multi_hand_landmarks:
#                             mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#             ret, buffer = cv2.imencode('.jpg', image)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                 b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
""" 19 april 2024"""
# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# """
# Determine Gestures is not needed, only for show at the moment.

# """

# def predict_gesture(landmarks):
#     # Implement your gesture recognition logic here using the hand landmarks
#     # This is a placeholder function, you'll need to write your own logic
#     if is_thumb_up(landmarks):
#         return "Thumb Up"
#     elif is_fist(landmarks):
#         return "Fist"
#     elif is_peace_sign(landmarks):
#         return "Peace Sign"
#     else:
#         return "Unknown"

# def is_thumb_up(landmarks):
#     # Check if the thumb is up
#     thumb_tip = landmarks[4]
#     thumb_mcp = landmarks[2]
#     index_tip = landmarks[8]

#     thumb_up = thumb_tip[1] < thumb_mcp[1] and thumb_tip[0] > index_tip[0]
#     return thumb_up

# def is_fist(landmarks):
#     # Check if the hand is in a fist
#     finger_tips = [landmarks[i] for i in [4, 8, 12, 16, 20]]
#     finger_closed = all(tip[1] > landmark[1] for tip, landmark in zip(finger_tips, [landmarks[i] for i in [3, 7, 11, 15, 19]]))
#     return finger_closed

# def is_peace_sign(landmarks):
#     # Check if the hand is in a peace sign
#     index_tip = landmarks[8]
#     middle_tip = landmarks[12]
#     index_mcp = landmarks[5]
#     middle_mcp = landmarks[9]

#     peace_sign = index_tip[1] < index_mcp[1] and middle_tip[1] < middle_mcp[1]
#     return peace_sign
# def gen_frames():
#     camera = cv2.VideoCapture(0)
#     mp_drawing = mp.solutions.drawing_utils
#     mp_hands = mp.solutions.hands
#     with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
#         while True:
#             success, frame = camera.read()
#             if not success:
#                 break
#             else:
#                 image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#                 # Flip the video so right hand displays on the right, and output is a mirror for the user
#                 image = cv2.flip(image, 1)
#                 image.flags.writeable = False
#                 results = hands.process(image)
#                 image.flags.writeable = True
#                 image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#                 # Detect the gesture using Mediapipe's hand landmark model
#                 if results.multi_hand_landmarks:
#                     for hand_landmarks in results.multi_hand_landmarks:
#                         mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#                         # Get the hand landmarks and predict the gesture
#                         landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark])
#                         gesture = predict_gesture(landmarks)

#                         # Display the gesture label on the video frame
#                         cv2.putText(image, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

#             ret, buffer = cv2.imencode('.jpg', image)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


if __name__ == '__main__':
    app.run(debug=True)

