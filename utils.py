import cv2
import numpy as np
import mediapipe as mp
from twilio.rest import Client
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers

sentence = []
actions = np.array(['hello','thanks','I love you','please','Yes','No','help','baby','Emergency','later'])

model = Sequential()
model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(30, 258),kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(LSTM(256, return_sequences=True, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(LSTM(128, return_sequences=False, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.load_weights('action_copy.h5')

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image,model):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,results

def draw_styled_landmarks(image,results):
    mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10),thickness=2,circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121),thickness=2,circle_radius=2))
    mp_drawing.draw_landmarks(image,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76),thickness=2,circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250),thickness=2,circle_radius=2))
    mp_drawing.draw_landmarks(image,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2))

colors = [(245,117,16),(117,245,16),(16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        color = colors[num % len(colors)]
        cv2.rectangle(output_frame, (0, 60 + num*40), (int(prob*100), 90 + num*40), color, -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num*40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

account_sid = 'AC91b775d11dd071b376477457f4175264'
auth_token = 'ac8cd707d40586229d1c56c46eb9eba5'
twilio_sms_number = '+12023359750' 
twilio_whatsapp_number = 'whatsapp:+14155238886'  

def validate_phone_number(phone_number):
    return bool(phone_number and phone_number.isdigit() and len(phone_number) == 10)

def send_message(method, phone_number, message):
    client = Client(account_sid, auth_token)
    
    if method.lower() == 'whatsapp':
        client.messages.create(body=message, from_=twilio_whatsapp_number, to=f'whatsapp:{phone_number}')
        return f"Message sent via WhatsApp to {phone_number}"
    elif method.lower() == 'sms':
        client.messages.create(body=message, from_=twilio_sms_number, to=phone_number)
        return f"Message sent via SMS to {phone_number}"
    
import nltk

nltk.download('punkt')

def actions_to_sentence(detected_actions):
    sentence = []
    
    if 'hello' in detected_actions:
        sentence.append('Hello!')
    
    if 'I love you' in detected_actions:
        sentence.append('I love you!')
    
    if 'thanks' in detected_actions:
        sentence.append('Thank you!')
    
    if 'help' in detected_actions:
        sentence.append('I need help!')
    
    if 'Emergency' in detected_actions:
        sentence.append('There is an emergency!')
    
    if 'later' in detected_actions:
        sentence.append('See you later!')

    if 'Yes' in detected_actions:
        sentence.append('Yes, that’s correct.')
    
    if 'No' in detected_actions:
        sentence.append('No, that’s not right.')
    
    if 'please' in detected_actions:
        sentence.append('Please, could you assist me?')

    if 'baby' in detected_actions:
        sentence.append('There’s a baby.')

    return sentence


def action_detection():
    global sentence
    sequence = []
    threshold = 0.95
    action_started = False
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)
            print(results)
            draw_styled_landmarks(image, results)
            if not action_started:
                if results.face_landmarks and results.left_hand_landmarks and results.right_hand_landmarks:
                    action_started = True
            if action_started:
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(actions[np.argmax(res)])
                    if res[np.argmax(res)] > threshold: 
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])
                    if len(sentence) > 5: 
                        sentence = sentence[-5:]
                    image = prob_viz(res, actions, image, colors)
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()
        cv2.destroyAllWindows()
        print(sentence)