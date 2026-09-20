# Real-Time ISL Alphabet Detector

This project is a real-time Indian Sign Language (ISL) alphabet detector. It uses your webcam to capture hand gestures, extracts 21 3D hand landmarks using MediaPipe, and classifies them into an alphabet (A-Z) using a lightweight Random Forest model.

The recognized text is accumulated into words and sentences and displayed as a live caption overlay at the bottom of the video feed.

## Why This Project?
- **Fast and Live Tracking**: By utilizing MediaPipe's lightweight hand tracking, the project achieves highly responsive, real-time hand detection and alphabet classification without needing heavy GPU resources.
- **Immediate Visual Feedback**: Once a hand is detected, the 21 key points are instantly mapped and tracked. The seamless transition from point-tracking to live letter and word detection creates a dynamic and interactive experience.
- **Practical Application (Accessibility)**: This acts as a bridge for communication. Integrating virtual camera support allows users to type subtitles in real-time during Zoom or Google Meet calls simply by signing, making online meetings more accessible.
- **Extensible Foundation**: The pipeline—from landmark extraction to classification—provides a solid foundation for adding more complex gestures or custom signs in the future.

## Technology Stack and Flow

**Tech Stack:**
- **OpenCV**: Used for capturing real-time video feed from the webcam and rendering visual overlays (bounding boxes, captions).
- **MediaPipe**: A framework developed by Google, utilized for its highly optimized and accurate hand tracking solution. It detects 21 3D hand landmarks in real-time.
- **Scikit-Learn (Random Forest)**: A robust machine learning library used to train a Random Forest Classifier on the extracted hand landmark data for recognizing the ISL alphabets.
- **NumPy & Pandas**: Essential for data manipulation and preprocessing the landmark coordinates into a structured format for training.
- **PyVirtualCam**: Allows the application output to be broadcasted as a virtual webcam for use in video conferencing tools.

**Project Flow:**
1. **Data Collection & Preprocessing**: Images of ISL signs are processed through MediaPipe to extract x, y, z coordinates of 21 hand landmarks. These are flattened and saved into a dataset (`data/landmarks.csv`).
2. **Model Training**: The Random Forest classifier is trained on this dataset to map specific landmark configurations to their corresponding alphabet (A-Z).
3. **Real-time Inference**: The application reads frames from the webcam. MediaPipe detects hands and extracts landmarks. These landmarks are instantly fed to the trained model to predict the alphabet.
4. **Captioning & Virtual Camera**: Recognized letters are smoothed out over a few frames (to avoid flickering), combined into words, and overlaid on the video. The final output is sent to both a local display window and a virtual camera.

## Features
- **Real-Time Landmark Extraction**: Fast and robust hand detection using MediaPipe.
- **Caption Overlay**: Smoothly accumulates recognized letters and displays them as a subtitle.
- **Virtual Camera Support**: Allows you to output the captioned feed into video call apps like Zoom or Google Meet.

## Setup

1. **Install Dependencies**
   Make sure you have Python 3.10+ installed.
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: pyvirtualcam might require additional backend software like OBS Virtual Camera on Windows to work).*

2. **Data Preprocessing**
   Extract landmarks from the provided dataset.
   ```bash
   cd src
   python data_preprocessing.py
   ```
   This will create a `data/landmarks.csv` file.

3. **Train the Model**
   Train the Random Forest classifier.
   ```bash
   cd src
   python train_model.py
   ```
   This will train the model and save it to `models/rf_model.pkl`.

## Usage

Run the live detector from the project root:
```bash
python main.py
```

### Controls:
- **Show your hand** to the camera. Hold a sign steady to "type" a letter.
- Hold no sign (or put your hand down) for a brief moment to type a **Space**.
- Press `b` to **Backspace** (delete the last letter).
- Press `c` to **Clear** the current text buffer.
- Press `q` to **Quit** the application.

### Virtual Camera Integration (Zoom / Google Meet)
If you want to use this as your camera source in a video call:
1. Ensure you have a virtual camera backend installed (e.g., OBS Virtual Cam).
2. Run the application with the `--virtual-cam` flag:
   ```bash
   python main.py --virtual-cam
   ```
3. Open Zoom or Google Meet and select **OBS Virtual Camera** (or whichever virtual camera was created) as your video input. You will now see your camera feed with live captions on the call!

## Project Structure
- `dataset_ISL/`: Original image dataset.
- `src/`: Source code for preprocessing, training, and real-time inference.
- `models/`: Trained models.
- `data/`: Extracted landmark features (CSV).
- `main.py`: Entry point for the application.

## Drawbacks & Challenges
- **Lack of Domain Expertise**: None of the project developers are natively familiar with Indian Sign Language. The data and nuances of actual signing may lack the fluidity or precision of a fluent signer.
- **Scaling to Word-Based Recognition**: The current scope is limited to static, alphabet-by-alphabet spelling. Evolving the project to recognize continuous, dynamic word-based gestures involves temporal modeling, which is significantly more complex.
- **Maintaining Accuracy at Scale**: Increasing the scope of the project (e.g., adding numbers, two-handed signs, or dynamic gestures) while keeping the same high level of accuracy and low latency can be very challenging.
