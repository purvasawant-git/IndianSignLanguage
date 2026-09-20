import os
import cv2
import json
import pandas as pd
import numpy as np
import argparse
from landmark_extractor import LandmarkExtractor

def rotate_image(image, angle_deg):
    """Rotates an image around its center with border replication."""
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def compute_canonical_dots(class_landmarks_list):
    """
    Computes the canonical (average) dot pattern of 21 hand landmarks across valid samples.
    First 63 values in the 182-dim vector are the normalized coordinates (x, y, z) * 21.
    """
    if not class_landmarks_list:
        return []
    
    arr = np.array(class_landmarks_list)
    # First 63 elements represent Hand 1 normalized (x, y, z) for 21 points
    hand1_coords = arr[:, :63]
    avg_coords = np.mean(hand1_coords, axis=0)
    
    DOT_NAMES = [
        "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
        "index_mcp", "index_pip", "index_dip", "index_tip",
        "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
        "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
        "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
    ]
    
    dots = []
    for i in range(21):
        x = round(float(avg_coords[i * 3]), 4)
        y = round(float(avg_coords[i * 3 + 1]), 4)
        z = round(float(avg_coords[i * 3 + 2]), 4)
        dots.append({
            "point_id": i,
            "name": DOT_NAMES[i],
            "x": x,
            "y": y,
            "z": z
        })
    return dots

def validate_coordinate_separability(class_patterns):
    """
    Validates coordinate data across all classes to identify any confusion/collisions.
    Returns collision warnings for any pairs with low separation distance.
    """
    warnings = []
    class_names = sorted(list(class_patterns.keys()))
    
    for i in range(len(class_names)):
        c1 = class_names[i]
        v1 = np.array(class_patterns[c1]['mean_vector'])
        for j in range(i + 1, len(class_names)):
            c2 = class_names[j]
            v2 = np.array(class_patterns[c2]['mean_vector'])
            
            # Euclidean distance between canonical coordinate profiles
            dist = float(np.linalg.norm(v1 - v2))
            if dist < 0.45:  # Closeness threshold in normalized coordinate space
                warnings.append({
                    "class_a": c1,
                    "class_b": c2,
                    "separation_distance": round(dist, 4),
                    "risk": "HIGH" if dist < 0.30 else "MODERATE"
                })
    return warnings

def main():
    parser = argparse.ArgumentParser(description="ISL Dot Pattern Extractor & Coordinate Validator")
    parser.add_argument('--max-per-class', type=int, default=120,
                        help="Max images to process per class (default: 120)")
    parser.add_argument('--dataset-dir', type=str, default='dataset_ISL',
                        help="Dataset directory path (default: dataset_ISL)")
    args = parser.parse_args()

    dataset_path = args.dataset_dir
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join('..', args.dataset_dir)
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset directory '{args.dataset_dir}' not found.")
            return

    # Initialize Landmark Extractor with static image mode and landmark validation
    extractor = LandmarkExtractor(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.25, enable_smoothing=False)
    
    # Dynamically find ALL class subfolders (alphabets a-z, numbers 0-9, or custom signs)
    classes = sorted([c for c in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, c))])
    
    data = []
    labels = []
    class_raw_landmarks = {}
    
    print(f"\n========================================================")
    print(f" [*] ISL DOT PATTERN LOCATOR & COORDINATE PROCESSOR")
    print(f" Target Classes ({len(classes)}): {', '.join([c.upper() for c in classes])}")
    print(f" Expandability: Fully compatible with numerical signs (0-9)")
    print(f"========================================================\n")
    
    total_images = 0
    extracted_images = 0
    
    for c in classes:
        class_path = os.path.join(dataset_path, c)
        files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if args.max_per_class > 0:
            files = files[:args.max_per_class]
            
        class_extracted = 0
        class_raw_landmarks[c.upper()] = []
        
        for f in files:
            img_path = os.path.join(class_path, f)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
                
            total_images += 1
            landmarks, _ = extractor.extract_landmarks(img, try_padding=True)
            
            if landmarks is not None:
                class_extracted += 1
                extracted_images += 1
                class_raw_landmarks[c.upper()].append(landmarks)
                
                # 1. Base sample
                data.append(landmarks)
                labels.append(c.upper())
                
                # 2. Horizontal mirror sample (flip left/right hand)
                flip_lms, _ = extractor.extract_landmarks(cv2.flip(img, 1), try_padding=True)
                if flip_lms is not None:
                    data.append(flip_lms)
                    labels.append(c.upper())
                    
                # 3. Hand tilt augmentations (+7° and -7°)
                tilt_pos, _ = extractor.extract_landmarks(rotate_image(img, 7.0), try_padding=True)
                if tilt_pos is not None:
                    data.append(tilt_pos)
                    labels.append(c.upper())

                tilt_neg, _ = extractor.extract_landmarks(rotate_image(img, -7.0), try_padding=True)
                if tilt_neg is not None:
                    data.append(tilt_neg)
                    labels.append(c.upper())

        samples_generated = class_extracted * 4
        print(f"  Class '{c.upper()}': {class_extracted}/{len(files)} images detected -> {samples_generated} samples stored.")

    print(f"\n========================================================")
    print(f" Feature Extraction Complete!")
    print(f" Processed: {extracted_images}/{total_images} images.")
    print(f" Total training samples generated: {len(data)}")
    print(f"========================================================\n")
    
    # Save CSV dataset
    csv_dir = '../data' if os.path.exists('../data') or os.path.exists('../src') else 'data'
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, 'landmarks.csv')
        
    df = pd.DataFrame(data)
    df['label'] = labels
    df.to_csv(csv_path, index=False)
    print(f"  [OK] Landmarks CSV saved successfully to: {csv_path}")

    # Build and Save Alphabet/Numerical Dot Patterns Database
    print("\nCalculating and saving canonical dot patterns per character...")
    patterns_db = {
        "metadata": {
            "total_classes": len(classes),
            "classes": [c.upper() for c in classes],
            "description": "Canonical 21-dot hand landmark patterns (expandable for alphabet A-Z and digits 0-9)"
        },
        "patterns": {}
    }
    
    validation_profiles = {}
    for c in classes:
        c_upper = c.upper()
        lms = class_raw_landmarks.get(c_upper, [])
        if lms:
            mean_vec = np.mean(np.array(lms), axis=0).tolist()
            canonical_dots = compute_canonical_dots(lms)
            patterns_db["patterns"][c_upper] = {
                "sample_count": len(lms) * 4,
                "dots_count": len(canonical_dots),
                "canonical_dots": canonical_dots
            }
            validation_profiles[c_upper] = {
                "mean_vector": mean_vec
            }

    patterns_json_path = os.path.join(csv_dir, 'patterns_database.json')
    with open(patterns_json_path, 'w') as f:
        json.dump(patterns_db, f, indent=2)
    print(f"  [OK] Canonical dot patterns saved to: {patterns_json_path}")

    # Coordinate Validation & Collision Check
    print("\nValidating coordinate separability across all characters...")
    collisions = validate_coordinate_separability(validation_profiles)
    if collisions:
        print(f"  [!] Coordinate Validation Notice: {len(collisions)} character pairs have close coordinate patterns:")
        for w in collisions[:8]:
            print(f"     - '{w['class_a']}' vs '{w['class_b']}': Distance = {w['separation_distance']} ({w['risk']} collision risk)")
    else:
        print("  [OK] Coordinate Validation Passed: Excellent separation across all characters!")

    print(f"\nPipeline ready for model training with pattern-aware validation.\n")

if __name__ == '__main__':
    main()
