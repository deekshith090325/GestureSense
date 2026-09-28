# GestureSense

Real-time Indian Sign Language (ISL) gesture recognition using MediaPipe hand
landmarks and a TensorFlow/Keras classifier.

## How It Works
1. **Landmark extraction:** MediaPipe detects the hand in each webcam frame and
   extracts hand landmarks (42 values per sample).
2. **Classification:** a TensorFlow/Keras model, [LSTM / dense neural network -
   write what your notebook actually uses], predicts the ISL gesture from the landmarks.
3. **Real-time output:** the predicted gesture is displayed live on the webcam feed.

## Project Structure
- `isl_detection.py`: main script for real-time ISL detection from webcam
- `dataset_keypoint_generation.py`: converts the ISL Kaggle image dataset into landmark features
- `keypoint.csv`: extracted landmark features for all images
- `ISL_classifier.ipynb`: notebook used to train and evaluate the classifier
- `model.h5`: trained classifier model

## Tech Stack
Python, MediaPipe, TensorFlow/Keras, OpenCV, NumPy, Pandas

## Setup
1. Clone the repo:
   `git clone https://github.com/deekshith090325/GestureSense`
2. Install dependencies:
   `pip install opencv-python mediapipe tensorflow numpy pandas`
3. Run real-time detection:
   `python isl_detection.py`

Press `q` to quit the webcam window. [check that this matches your script]

## Results
- Test accuracy: [xx%, copy from the last cell of ISL_classifier.ipynb]
- Number of gesture classes: [n]

## Limitations & Future Work
- Small dataset; more samples per gesture would improve accuracy
- Currently shows the predicted gesture as a label; text/speech output and a GUI
  are not built yet
- Support for more ISL gestures
