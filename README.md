# GestureSense

Real-time Indian Sign Language (ISL) gesture recognition using MediaPipe hand
landmarks and a dense neural network classifier.

## How It Works
1. **Landmark extraction:** MediaPipe detects the hand in each frame and extracts
   42 hand-landmark features per sample.
2. **Classification:** a fully-connected (dense) neural network — 4 hidden layers
   (1470 → 832 → 428 → 264 units, ReLU, with Dropout) and a softmax output layer —
   classifies the landmarks into one of 3 ISL gesture classes (A, B, C).
3. **Real-time output:** the predicted gesture is displayed live on the webcam feed.

## Project Structure
- `isl_detection.py`: main script for real-time ISL detection from webcam
- `dataset_keypoint_generation.py`: converts the ISL Kaggle image dataset into landmark features
- `keypoint.csv`: extracted landmark features for all images
- `ISL_classifier.ipynb`: notebook used to train and evaluate the classifier
- `model.h5`: trained classifier model

## Tech Stack
Python, MediaPipe, TensorFlow/Keras, OpenCV, NumPy, Pandas, scikit-learn

## Setup
1. Clone the repo:
   `git clone https://github.com/deekshith090325/GestureSense`
2. Install dependencies:
   `pip install -r requirements.txt`
3. Run real-time detection:
   `python isl_detection.py`

## Results
Achieved 100% accuracy, precision, recall, and F1-score on the held-out test
split (3 gesture classes).

## Limitations & Future Work
- Currently supports only 3 gesture classes (A, B, C); needs a full ISL alphabet
- Small dataset; a larger and more varied dataset would improve robustness
- Displays the predicted gesture as a label only — text/speech output and a GUI
  are not built yet
