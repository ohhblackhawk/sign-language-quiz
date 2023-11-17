# app.py

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    # Add your image processing logic here
    # For simplicity, let's assume the image is sent as a file in a form
    uploaded_file = request.files['file']
    # Process the image using your machine learning model
    result = "Sign language detected!"
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
