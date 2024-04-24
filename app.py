# app.py
from flask import Flask, render_template, request, make_response, send_file,redirect, url_for
import cv2
import mediapipe as mp 
import re
from keras.models import load_model
import numpy as np
from flask_socketio import SocketIO, emit


app = Flask(__name__)

#this should be stored else where, just placing it here for now.
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

#load model
cnn_model = load_model('cnn_model.h5')
#num to letter
label_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

#sockets
@socketio.on('connect')
def connect():
    print('Client connected')

@socketio.on('disconnect')
def disconnect():
    print('Client disconnected')

@socketio.on('image')
def process_image(blob):
    try:
        # Convert blob to NumPy array
        frame = np.frombuffer(blob, np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        if frame is None:
            print("Error: Unable to decode image")
            socketio.emit('error', 'Error processing image')
            return

        #process the image frame using mediapiep n cnn model
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=True,max_num_hands = 2, min_detection_confidence = 0.5)
        results = hands.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        print('processed hand detection')

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_list = normalise_landmark(hand_landmarks)
                predictions = get_predictions_from_model(landmark_list, cnn_model)
                print("Predicted letter:", predictions)
                socketio.emit('prediction',predictions)
                print("Emitted prediction to client")
    except cv2.error as e:
        print(f"Error processing image: {e}")
        socketio.emit('error', 'Error processing image')


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

# choose difficulty
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
easy_word_pool = ['a','b','c','d']
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
    

if __name__ == '__main__':
    # app.run(debug=True)
    socketio.run(app, host='localhost', port=5000)

