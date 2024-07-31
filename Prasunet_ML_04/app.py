import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

model = load_model('hand_gesture_Model.h5')

def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (224, 224)) 
    frame_array = np.array(resized_frame, dtype=np.float32)
    frame_array /= 255.0
    frame_array = np.expand_dims(frame_array, axis=0)
    return frame_array

cap = cv2.VideoCapture(0)

gesture_labels = ["open palm", "closed palm", "index", "Thumb", "fist","ok", "Thumbs Up", "Thumbs Down", "OK Sign", "Peace Sign", "Fist",
    "Open Palm", "Pointing", "Victory Sign", "Stop Sign", "Rock On", "Call Me", "Clap", "Finger Gun", "Crossed Fingers", "Heart Sign"] 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    preprocessed_frame = preprocess_frame(frame)
    
    predictions = model.predict(preprocessed_frame)
    predicted_label = gesture_labels[np.argmax(predictions)]

    cv2.putText(frame, f'Prediction: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Hand Gesture Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
