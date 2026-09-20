import cv2
import numpy as np
import time

class TextBuffer:
    def __init__(self, hold_target_seconds=1.0, space_timeout=1.2):
        """
        Initializes TextBuffer with a 1.0-second dwell timer before confirming letters.
        - hold_target_seconds: Time (default 1.0s) a user must hold a sign before it types.
        - space_timeout: Inactivity time (default 1.2s) without hands before adding a space.
        """
        self.hold_target_seconds = hold_target_seconds
        self.space_timeout = space_timeout
        self.current_word = ""
        self.sentence = ""
        
        # Hold state
        self.current_candidate = None
        self.hold_start_time = None
        self.hold_progress = 0.0
        self.hold_elapsed = 0.0
        self.last_confirmed_letter = None
        self.confirmation_flash_time = 0.0
        self.last_hand_seen_time = time.time()
        
    def process_prediction(self, predicted_letter):
        """
        Feeds the live predicted letter into the 2-3 second hold timer.
        Returns:
            dict: Current hold status for overlay rendering.
        """
        now = time.time()
        just_typed = None

        if predicted_letter is not None:
            self.last_hand_seen_time = now

            # If same candidate as current hold
            if predicted_letter == self.current_candidate:
                self.hold_elapsed = now - self.hold_start_time
                self.hold_progress = min(1.0, self.hold_elapsed / self.hold_target_seconds)

                # Has the user held the sign for the full 2-3 seconds?
                if self.hold_elapsed >= self.hold_target_seconds:
                    # Append letter to current word
                    self.current_word += predicted_letter
                    self.last_confirmed_letter = predicted_letter
                    self.confirmation_flash_time = now
                    just_typed = predicted_letter
                    
                    # Reset hold timer for next character
                    self.current_candidate = None
                    self.hold_start_time = None
                    self.hold_progress = 0.0
                    self.hold_elapsed = 0.0
            else:
                # Started holding a new gesture
                self.current_candidate = predicted_letter
                self.hold_start_time = now
                self.hold_elapsed = 0.0
                self.hold_progress = 0.0
        else:
            # No hand detected or ambiguous gesture
            self.current_candidate = None
            self.hold_start_time = None
            self.hold_progress = 0.0
            self.hold_elapsed = 0.0

            # Auto-space check: if hands lowered for 2-3 seconds, commit word with trailing space
            if len(self.current_word) > 0 and (now - self.last_hand_seen_time) > self.space_timeout:
                self.sentence += self.current_word + " "
                self.current_word = ""
                self.last_hand_seen_time = now

        return {
            "candidate": self.current_candidate,
            "progress": self.hold_progress,
            "elapsed": self.hold_elapsed,
            "target": self.hold_target_seconds,
            "just_typed": just_typed,
            "is_flashing": (now - self.confirmation_flash_time) < 0.8
        }

    def backspace(self):
        """Deletes the last character."""
        if len(self.current_word) > 0:
            self.current_word = self.current_word[:-1]
        elif len(self.sentence) > 0:
            self.sentence = self.sentence.rstrip()
            if len(self.sentence) > 0:
                self.sentence = self.sentence[:-1]

    def clear(self):
        """Clears all typed text."""
        self.current_word = ""
        self.sentence = ""
        self.current_candidate = None
        self.hold_progress = 0.0

    def get_display_text(self):
        """Returns the full accumulated sentence and current word."""
        return self.sentence + self.current_word

def draw_overlay(frame, display_text, current_pred, bbox=None, hold_info=None):
    """
    Renders live caption overlay, bounding box, and 2-3s hold progress bar.
    """
    h, w, _ = frame.shape
    
    # 1. Draw hand bounding box & live hold progress
    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        
        # Color: Green if holding/confirmed, Orange if idle
        is_flashing = hold_info and hold_info.get("is_flashing", False)
        box_color = (0, 255, 0) if is_flashing else (0, 215, 255)
        
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
        
        # Text label above hand
        if hold_info and hold_info.get("is_flashing"):
            label_str = f"LOCKED IN: '{display_text[-1] if display_text else ''}'"
            cv2.putText(frame, label_str, (x_min, max(28, y_min - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        elif hold_info and hold_info.get("candidate"):
            c = hold_info["candidate"]
            el = hold_info["elapsed"]
            tgt = hold_info["target"]
            label_str = f"Hold '{c}': {el:.1f}s / {tgt:.1f}s"
            cv2.putText(frame, label_str, (x_min, max(28, y_min - 22)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
            # Live Progress Bar right above hand
            bar_w = max(120, x_max - x_min)
            bar_h = 8
            bar_x = x_min
            bar_y = max(10, y_min - 12)
            prog = hold_info["progress"]
            
            # Background track (dark grey)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
            # Filled progress (bright green)
            fill_w = int(bar_w * prog)
            if fill_w > 0:
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 255, 0), -1)
            # Border
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)
        elif current_pred is not None:
            cv2.putText(frame, f"Sign: {current_pred}", (x_min, max(28, y_min - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # 2. Draw Bottom Subtitle Bar (HUD)
    overlay = frame.copy()
    bar_height = 85
    cv2.rectangle(overlay, (0, h - bar_height), (w, h), (18, 18, 18), -1)
    alpha = 0.80
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Top accent line
    cv2.line(frame, (0, h - bar_height), (w, h - bar_height), (0, 200, 255), 2)

    # Line 1: Accumulated Text / Word
    typed_str = f"Typed: \"{display_text}\"" if display_text else "Typed: (Show hand gesture & hold 1s to type)"
    cv2.putText(frame, typed_str, (25, h - 48), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Line 2: Real-time Status / Hold Timer Indicator
    if hold_info and hold_info.get("candidate"):
        cand = hold_info["candidate"]
        prog_pct = int(hold_info["progress"] * 100)
        el = hold_info["elapsed"]
        tgt = hold_info["target"]
        
        # Render clean ASCII progress bar in bottom bar
        total_blocks = 12
        filled_blocks = int(total_blocks * hold_info["progress"])
        bar_visual = "=" * filled_blocks + "-" * (total_blocks - filled_blocks)
        status_str = f"Holding '{cand}' [{bar_visual}] {el:.1f}s / {tgt:.1f}s ({prog_pct}%)"
        status_color = (0, 255, 0) if prog_pct >= 80 else (0, 255, 255)
    elif hold_info and hold_info.get("is_flashing"):
        status_str = "[CONFIRMED] Letter added! Switch to next sign."
        status_color = (0, 255, 0)
    elif current_pred:
        status_str = f"Detected: {current_pred} (hold steady 1s to type)"
        status_color = (200, 200, 200)
    else:
        status_str = "Drop hands for 1.2s to add space  |  'l'=Points  'b'=Backspace  'c'=Clear  'q'=Quit"
        status_color = (150, 150, 150)

    cv2.putText(frame, status_str, (25, h - 18), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)

    return frame
