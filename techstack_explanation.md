# Basic Tech Stack Explanation

This document provides a simple breakdown of the technologies used in this project (Live Indian Sign Language Recognition) and their primary purposes.

## 1. Core Language
- **Python**: The main programming language used to tie all the libraries together and write the application logic.

## 2. Computer Vision & Tracking
- **OpenCV (`opencv-python`)**: 
  - **What it is**: A powerful computer vision library.
  - **What it's for**: Used for capturing video from your webcam, processing those video frames (resizing, converting colors), and drawing bounding boxes or text on the screen.
- **MediaPipe (`mediapipe`)**: 
  - **What it is**: A framework built by Google for building multimodal applied machine learning pipelines.
  - **What it's for**: Used for real-time tracking of hand landmarks, body pose, and facial features. It extracts the exact coordinates of joints/fingertips from the camera feed which are essential for recognizing sign language gestures.

## 3. Data Processing & Machine Learning
- **NumPy (`numpy`)**: 
  - **What it is**: A library for numerical computations.
  - **What it's for**: Used for handling the coordinates and feature arrays efficiently. When MediaPipe extracts landmark data, NumPy helps in reshaping and structuring this data for the machine learning model.
- **Pandas (`pandas`)**:
  - **What it is**: A data manipulation and analysis library.
  - **What it's for**: Likely used in the data preprocessing phase (`data_preprocessing.py`) to organize extracted features into a structured format (like CSV) and manage datasets for training.
- **Scikit-learn (`scikit-learn`)**:
  - **What it is**: A popular machine learning library in Python.
  - **What it's for**: Used for building and training the actual gesture recognition model. Instead of deep learning, Scikit-learn allows the training of fast, lightweight classifiers (like Support Vector Machines, Random Forests, or Logistic Regression) based on the landmark data provided by MediaPipe.

## 4. Output & Integration
- **PyVirtualCam (`pyvirtualcam`)**:
  - **What it is**: A library for sending frames to a virtual camera.
  - **What it's for**: As seen in `virtual_cam_output.py`, this allows the processed video feed (with sign language translations overlayed) to be outputted as a "Virtual Webcam." This means you can use your sign language translation system live in video conferencing apps like Zoom, Google Meet, or MS Teams.
