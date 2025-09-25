from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('emotion_model.h5')

# Define emotion labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_emotion(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Extract face ROI
        roi = gray[y:y+h, x:x+w]
        
        # Resize to 48x48
        roi = cv2.resize(roi, (48, 48))
        
        # Normalize
        roi = roi / 255.0
        
        # Reshape for model input
        roi = np.reshape(roi, (1, 48, 48, 1))
        
        # Predict emotion
        prediction = model.predict(roi)
        emotion_idx = np.argmax(prediction[0])
        emotion = emotions[emotion_idx]
        
        # Draw rectangle and emotion text
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    return frame

def generate_frames():
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Process frame for emotion detection
        processed_frame = detect_emotion(frame)
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True) 