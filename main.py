import argparse
import sys
import os

# Add src to python path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from realtime_inference import RealtimeInference

def main():
    parser = argparse.ArgumentParser(description="Live ISL Alphabet & Numerical Gesture Detector")
    parser.add_argument('--virtual-cam', action='store_true', 
                        help="Output to a virtual camera (e.g. for Zoom/Meet)")
    parser.add_argument('--model-path', type=str, default='models/rf_model.pkl',
                        help="Path to the trained model pickle file")
    parser.add_argument('--camera-index', type=int, default=0,
                        help="Webcam device index (default: 0)")
    parser.add_argument('--hold-time', type=float, default=1.0,
                        help="Hold duration in seconds required before confirming a letter (default: 1.0s)")
                        
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        print("Please train the model first by running: python src/train_model.py")
        return
        
    inference = RealtimeInference(
        model_path=args.model_path, 
        use_virtual_cam=args.virtual_cam,
        hold_time=args.hold_time
    )
    inference.run(camera_index=args.camera_index)

if __name__ == '__main__':
    main()
