# app.py
from flask import Flask, render_template, request, make_response, send_file,redirect, url_for, abort
import cv2
import mediapipe as mp 
import re
from keras.models import load_model
import numpy as np
from flask_socketio import SocketIO, emit
import random


app = Flask(__name__)

#this should be stored else where, just placing it here for now.
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

#load model
cnn_model = load_model('cnn_model.h5')
#num to letter
label_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
#global 
#word pool
easy_word_pool = ['cat', 'dog', 'hut', 'fudge', 'boy', 'god', 'yes', 'tower']
medium_word_pool = ['house', 'cloud', 'dance', 'flute', 'space', 'horse', 'knife', 'image', 'fruit', 'stamp']
hard_word_pool = ['jacket', 'ranger', 'jungle', 'zebra','earth']
#thequickbrownfoxjumpsoverlazydog
custom_word_pool = []


current_word = None
current_index = 0

#sockets
@socketio.on('connect')
def connect():
    global current_word
    initial_letter = current_word[0] 
    emit('update_letter', initial_letter, broadcast=True)

@socketio.on('disconnect')
def disconnect():
    print('Client disconnected')

@socketio.on('image')
def handle_image(blob):
    global current_index
    global current_word
    predicted_letter = predict_letter(blob)
    if predicted_letter:
        #word fully spelt?
        if current_index == len(current_word):
            emit('word_spelt', 'Well done!', broadcast=True)
            emit('redirect', url_for('difficulty'), broadcast=True)
            return
        #predicted letter matches letter?
        if predicted_letter == current_word[current_index]:
            current_index += 1
            #valid range
            if 0 <= current_index < len(current_word):
                #update on client side
                emit('update_letter', current_word[current_index], broadcast=True)
            else:
                #goes out of index
                pass
            #checks word has been fully spelled
            if current_index == len(current_word):
                emit('word_spelt', 'Well done!', broadcast=True)
                emit('redirect', url_for('difficulty'), broadcast=True)
        #update predict letter on client side
        emit('prediction', predicted_letter, broadcast=True)

def predict_letter(blob):
    try:
        #blob to numpy array
        frame = np.frombuffer(blob, np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        if frame is None:
            print("Error: Unable to decode image")
            return None
        #mediapipe and cnn model
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                #norlaise landmarks
                normalised_landmarks = normalise_landmark(hand_landmarks)
                #get prediction
                predicted_letter = get_predictions_from_model(normalised_landmarks, cnn_model)
                return predicted_letter
        return None
    except cv2.error as e:
        print(f"Error processing image: {e}")
        return None
    
#prediction
def get_predictions_from_model(landmark_list, cnn_model):
    #predicts ^-^
    predictions = cnn_model.predict(landmark_list)
    predicted_class = np.argmax(predictions)
    #letters from class
    alphabetical_label = label_names[predicted_class]
    return alphabetical_label

#normalise hand landmarks
def normalise_landmark(hand_landmarks):
    #relative
    landmark_list = [[hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y] for i in range(len(hand_landmarks.landmark))]
    #flattens list
    landmark_list = [x for pair in landmark_list for x in pair]
    #convert to relative coordinates, where origin is first landmark
    base_x, base_y = landmark_list[0], landmark_list[1]
    landmark_list = [(x - base_x) for x in landmark_list[::2]] + [(y - base_y) for y in landmark_list[1::2]]
    #maximum absolute value
    max_value = max(list(map(abs, landmark_list)))
    #normalise 
    landmark_list = [n / max_value for n in landmark_list]
    #reshape landmarks for model input (same way it was trained)
    landmark_list = np.array(landmark_list).reshape((1, 42, 1))
    return landmark_list
    
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/signs')
def signs():
    return render_template('signs.html')

#caches images
@app.route('/app/static/images/<letter>.gif')
def send_image(letter):
    #cache timeout 1 hr
    cache_timeout = 3600
    #attaching headers, to store the sign gifs
    response = make_response(send_file('/static/images/{}.gif'.format(letter)))
    response.headers['Cache-Control'] = 'max-age={}, public'.format(cache_timeout)
    return response

#difficulty
@app.route('/difficulty')
def difficulty():
    return render_template('difficulty.html')

@app.route('/quiz/<difficulty>')
def quiz(difficulty):
    global current_word, current_index
    if difficulty == 'easy':
        word_pool = easy_word_pool
    elif difficulty == 'medium':
        word_pool = medium_word_pool
    elif difficulty == 'hard':
        word_pool = hard_word_pool
    elif difficulty == 'custom':
        word_pool = custom_word_pool
    else:
        return render_template('difficulty.html', error_message="Invalid difficulty level!")

    if not word_pool:
        return render_template('difficulty.html', error_message="No words in this difficulty level!")

    current_word = random.choice(word_pool)
    current_index = 0
    return render_template('quiz.html', word_pool=word_pool, difficulty=difficulty, current_word=current_word, current_index=current_index)

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



if __name__ == '__main__':
    # app.run(debug=True)
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True, debug=True)

