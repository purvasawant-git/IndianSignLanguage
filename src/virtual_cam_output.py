import pyvirtualcam
import cv2

class VirtualCamWrapper:
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.cam = None
        try:
            self.cam = pyvirtualcam.Camera(width=self.width, height=self.height, fps=self.fps)
            print(f"Using virtual camera: {self.cam.device}")
        except Exception as e:
            print(f"Could not start virtual camera (is OBS or similar installed?): {e}")

    def send_frame(self, frame):
        if self.cam is not None:
            # OpenCV is BGR, pyvirtualcam expects RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.cam.send(frame_rgb)
            self.cam.sleep_until_next_frame()

    def close(self):
        if self.cam is not None:
            self.cam.close()
