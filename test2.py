import sys
import mediapipe
print("MediaPipe version:", getattr(mediapipe, '__version__', 'unknown'))
try:
    import mediapipe.python.solutions.hands as mp_hands
    print("Found hands via python.solutions")
except Exception as e:
    print("Failed python.solutions:", e)

try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    print("Found tasks API")
except Exception as e:
    print("Failed tasks API:", e)
