import cv2
import pickle
import numpy as np
import time
from landmark_extractor import LandmarkExtractor
from caption_overlay import TextBuffer, draw_overlay
from virtual_cam_output import VirtualCamWrapper

class RealtimeInference:
    def __init__(self, model_path='models/rf_model.pkl', use_virtual_cam=False, hold_time=1.0):
        """
        Realtime ISL inference engine with 1.0s hold-to-type dwell timer.
        - hold_time: seconds required to hold a sign before confirming (default: 1.0s)
        """
        self.extractor = LandmarkExtractor(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.35, enable_smoothing=True)
        self.text_buffer = TextBuffer(hold_target_seconds=hold_time, space_timeout=1.2)
        
        # Load Model
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.le = data['label_encoder']
            
        self.use_virtual_cam = use_virtual_cam
        self.vcam = None
        self.hold_time = hold_time
        self.smoothed_probs = None
        self.show_landmarks = True

    def run(self, camera_index=0):
        # Try DirectShow first on Windows for lowest latency, fallback to default backend
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)
            
        if not cap.isOpened():
            print(f"Error: Could not open webcam (index {camera_index}).")
            print("Check that no other app (Zoom, Teams, Chrome, Camera) is using the webcam.")
            return

        # Warm up camera
        time.sleep(0.2)
        
        # Setup virtual cam if requested
        if self.use_virtual_cam:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            if fps == 0: fps = 30
            self.vcam = VirtualCamWrapper(width=width, height=height, fps=fps)

        print("\n========================================================")
        print(" [*] Live ISL Alphabet Detector Active")
        print(f" Hold Duration: {self.hold_time:.1f}s required before typing each letter")
        print(" Space Timer:   Drop hands for 1.2s to insert a space")
        print(" Hotkeys:       'b' = Backspace | 'c' = Clear text | 'l' = Toggle Landmarks | 'q' = Quit")
        print("========================================================\n")
        
        consecutive_failures = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures > 30:
                    print("\nError: Lost video feed from webcam.")
                    break
                time.sleep(0.05)
                continue
            
            consecutive_failures = 0
            frame = cv2.flip(frame, 1) # Mirror image for natural hand movement
            
            # Fast extraction with temporal landmark smoothing for stable points
            landmarks, hand_objs = self.extractor.extract_landmarks(frame, try_padding=False)
            current_pred = None
            candidate_letter = None
            bbox = None
            
            if landmarks is not None:
                X_input = np.array(landmarks).reshape(1, -1)
                
                # Check confidence and pattern distinctiveness using smoothed probabilities
                if hasattr(self.model, 'predict_proba'):
                    raw_probs = self.model.predict_proba(X_input)[0]
                    if self.smoothed_probs is None:
                        self.smoothed_probs = raw_probs
                    else:
                        # 65% history + 35% current frame -> completely eliminates frame-to-frame fluctuation
                        self.smoothed_probs = 0.65 * self.smoothed_probs + 0.35 * raw_probs
                        
                    probs = self.smoothed_probs
                    top_indices = np.argsort(probs)[::-1]
                    top1_idx, top2_idx = top_indices[0], top_indices[1]
                    top1_prob, top2_prob = probs[top1_idx], probs[top2_idx]
                    margin = top1_prob - top2_prob
                    
                    # Confirm stable pattern if confidence and separation margin are met
                    if top1_prob > 0.22 and margin >= 0.05:
                        pred_label = self.le.inverse_transform([top1_idx])[0]
                        candidate_letter = pred_label.upper()
                        current_pred = f"{candidate_letter} ({int(top1_prob * 100)}%)"
                    elif top1_prob > 0.18:
                        # Ambiguous pattern (colliding with another letter)
                        cand1 = self.le.inverse_transform([top1_idx])[0].upper()
                        cand2 = self.le.inverse_transform([top2_idx])[0].upper()
                        current_pred = f"{cand1}? ({cand2})"
                        candidate_letter = None
                else:
                    pred_idx = self.model.predict(X_input)[0]
                    pred_label = self.le.inverse_transform([pred_idx])[0]
                    candidate_letter = pred_label.upper()
                    current_pred = candidate_letter
            else:
                self.smoothed_probs = None
                    
            # Compute bounding box & render hand tracking points across ALL detected hands using OpenCV
            if hand_objs:
                h, w, _ = frame.shape
                x_coords = []
                y_coords = []
                for hand in hand_objs:
                    x_coords.extend([lm.x for lm in hand])
                    y_coords.extend([lm.y for lm in hand])
                
                if x_coords and y_coords:
                    x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
                    y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
                    
                    # Add padding
                    padding = 20
                    bbox = (max(0, x_min - padding), max(0, y_min - padding), 
                            min(w, x_max + padding), min(h, y_max + padding))
                            
                # Draw visible smoothed hand landmarks and connection lines on OpenCV frame
                if self.show_landmarks:
                    for i, hand in enumerate(hand_objs):
                        hand_label = f"Hand {i+1}" if len(hand_objs) > 1 else None
                        self.extractor.draw_landmarks(frame, hand, hand_label=hand_label)
            
            # Feed candidate into 2-3s hold-to-type timer
            hold_info = self.text_buffer.process_prediction(candidate_letter)
            
            # Draw HUD overlay with hold progress bar and subtitles
            display_text = self.text_buffer.get_display_text()
            frame = draw_overlay(frame, display_text, current_pred, bbox, hold_info)
            
            # Output display
            cv2.imshow("ISL Alphabet Detector", frame)
            if self.vcam:
                self.vcam.send_frame(frame)
                
            # Keyboard Inputs
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.text_buffer.clear()
            elif key == ord('b'):
                self.text_buffer.backspace()
            elif key == ord('l'):
                self.show_landmarks = not self.show_landmarks
                status = "ON" if self.show_landmarks else "OFF"
                print(f" [*] Hand tracking points: {status}")

        cap.release()
        cv2.destroyAllWindows()
        if self.vcam:
            self.vcam.close()
