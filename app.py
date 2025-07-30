from utils import *
from flask import Flask, render_template, Response, jsonify, request

app = Flask(__name__)

@app.route('/video_feed')
def video_feed():
    return Response(action_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_sentence')
def get_sentence():
    return jsonify(sentence=sentence)

@app.route('/validate_number', methods=['POST'])
def validate_number():
    data = request.get_json()
    phone_number = data.get("phoneNumber") if data else None
    if validate_phone_number(phone_number):
        return jsonify(True)
    else:
        return jsonify(False)

@app.route('/send_message', methods=['POST'])
def send_messages():
    global sentence
    data = request.get_json()
    if data and 'method' in data and 'phoneNumber' in data:
        final_sentence = actions_to_sentence(sentence)
        final_sentence_text = ' '.join(final_sentence)
        response = send_message(data['method'], data['phoneNumber'], final_sentence_text)
        return jsonify({"status": "success", "message": "Message sent successfully!"}), 200
    else:
        return jsonify({"status": "error", "message": "Invalid data"}), 400
    
@app.route('/', methods=['GET'])
def hello():
    return render_template('index.html')