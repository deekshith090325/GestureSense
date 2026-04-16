import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import string
from tensorflow import keras
from collections import deque

# Load model
model = keras.models.load_model("model.h5", compile=False)

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Alphabet labels (adjust if your training used different labels)
alphabet = ['1','2','3','4','5','6','7','8','9']
alphabet += list(string.ascii_uppercase)

# Prediction smoothing
prediction_history = deque(maxlen=10)

# Webcam
cap = cv2.VideoCapture(0)

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = np.array(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    temp_landmark_list = temp_landmark_list - [base_x, base_y]

    # Flatten
    temp_landmark_list = temp_landmark_list.flatten()

    # Normalize
    max_value = max(list(map(abs, temp_landmark_list)))
    if max_value == 0:
        return []

    temp_landmark_list = temp_landmark_list / max_value

    return temp_landmark_list.tolist()

while True:
    ret, image = cap.read()
    if not ret:
        break

    # Flip for correct orientation
    image = cv2.flip(image, 1)

    debug_image = image.copy()

    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    label = "No Hand"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # Get landmarks
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)

            # Preprocess
            pre_processed_landmark_list = pre_process_landmark(landmark_list)

            # Safety check
            if pre_processed_landmark_list is None or len(pre_processed_landmark_list) == 0:
                continue

            # Convert to model input
            input_data = np.array(pre_processed_landmark_list)
            input_data = input_data.reshape(1, -1)

            # Predict
            predictions = model.predict(input_data, verbose=0)

            predicted_class = np.argmax(predictions)
            confidence = np.max(predictions)

            # Threshold
            if confidence < 0.6:
                current_prediction = "Unknown"
            else:
                current_prediction = alphabet[predicted_class]

            # Smooth predictions
            prediction_history.append(current_prediction)
            label = max(set(prediction_history), key=prediction_history.count)

            # Draw landmarks
            mp_drawing.draw_landmarks(
                debug_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # Display label
    cv2.putText(debug_image, f"{label}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show window
    cv2.imshow("ISL Detection", debug_image)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()