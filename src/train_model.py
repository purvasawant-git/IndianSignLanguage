import pandas as pd
import numpy as np
import os
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def compute_coordinate_separation_report(X, y, class_names):
    """
    Validates coordinate data across all characters:
    - Computes centroid of landmark coordinates for each character.
    - Measures pairwise Euclidean separation distance.
    - Identifies any character pairs that have high risk of confusion.
    """
    centroids = {}
    for idx, c in enumerate(class_names):
        mask = (y == idx)
        if np.any(mask):
            centroids[c] = np.mean(X[mask], axis=0)

    report = {
        "total_classes": len(class_names),
        "classes": class_names.tolist() if hasattr(class_names, 'tolist') else list(class_names),
        "collision_risks": [],
        "min_separation_distance": 999.0,
        "closest_pair": None
    }

    names = list(centroids.keys())
    for i in range(len(names)):
        c1 = names[i]
        for j in range(i + 1, len(names)):
            c2 = names[j]
            dist = float(np.linalg.norm(centroids[c1] - centroids[c2]))
            if dist < report["min_separation_distance"]:
                report["min_separation_distance"] = round(dist, 4)
                report["closest_pair"] = f"{c1} vs {c2}"
            if dist < 0.65:
                report["collision_risks"].append({
                    "char_1": c1,
                    "char_2": c2,
                    "separation_dist": round(dist, 4),
                    "status": "WATCHLIST" if dist < 0.45 else "SEPARABLE"
                })
    
    return report

def train():
    data_path = '../data/landmarks.csv'
    if not os.path.exists(data_path):
        data_path = 'data/landmarks.csv'
        if not os.path.exists(data_path):
            print("Data file 'landmarks.csv' not found. Run data_preprocessing.py first.")
            return

    print("\n========================================================")
    print(" [*] ISL MODEL TRAINING & COORDINATE VALIDATION")
    print(f" Source: {data_path}")
    print("========================================================\n")

    df = pd.read_csv(data_path)
    X = df.drop('label', axis=1).values
    y_raw = df['label'].astype(str).str.upper().values
    
    classes_found = np.unique(y_raw)
    print(f"Loaded {len(X)} samples with {X.shape[1]} features across {len(classes_found)} classes:")
    print(f"Classes: {', '.join(sorted(classes_found))}\n")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    class_names = [str(c).upper() for c in le.classes_]
    
    # Step 1: Validate coordinate data and measure character separation
    print("Step 1: Validating coordinate distinctiveness to prevent character confusion...")
    coord_report = compute_coordinate_separation_report(X, y_encoded, class_names)
    print(f"  [OK] Minimum class separation distance: {coord_report['min_separation_distance']} ({coord_report['closest_pair']})")
    
    if coord_report["collision_risks"]:
        print(f"  [!] Notice: {len(coord_report['collision_risks'])} character pairs have similar hand shapes:")
        for r in coord_report["collision_risks"][:6]:
            print(f"     - '{r['char_1']}' vs '{r['char_2']}': distance = {r['separation_dist']} ({r['status']})")
    else:
        print("  [OK] All character coordinate patterns are widely separated and distinct!")

    # Save coordinate validation report
    report_dir = '../data' if os.path.exists('../data') else 'data'
    report_path = os.path.join(report_dir, 'coordinate_validation_report.json')
    with open(report_path, 'w') as f:
        json.dump(coord_report, f, indent=2)
    print(f"  [OK] Saved validation report to: {report_path}\n")

    # Step 2: Stratified Train / Test Split
    print("Step 2: Splitting dataset into training (80%) and validation (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Step 3: Train Random Forest with balanced weights
    print("Step 3: Training Random Forest (200 trees, balanced weights, multi-core)...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    # Step 4: Evaluate test performance & check confusion matrix
    print("Step 4: Evaluating accuracy on holdout test set...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n========================================================")
    print(f" [*] Final Test Accuracy: {acc * 100:.2f}%")
    print("========================================================\n")
    
    cm = confusion_matrix(y_test, y_pred)
    actual_collisions = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i][j] > 0:
                actual_collisions.append((class_names[i], class_names[j], int(cm[i][j])))
                
    if actual_collisions:
        actual_collisions.sort(key=lambda x: x[2], reverse=True)
        print("Confusion Matrix Check:")
        for c1, c2, count in actual_collisions[:8]:
            print(f"  - '{c1}' confused with '{c2}': {count} test cases")
    else:
        print(" [OK] Zero collisions on test set! Perfect separation achieved across all characters.")
        
    print("\nDetailed Classification Summary:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Step 5: Save model and label encoder
    model_dir = '../models' if os.path.exists('../models') or os.path.exists('../src') else 'models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'rf_model.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump({'model': clf, 'label_encoder': le}, f)
        
    print(f" [OK] Model and label encoder successfully saved to: {model_path}\n")

if __name__ == '__main__':
    train()
