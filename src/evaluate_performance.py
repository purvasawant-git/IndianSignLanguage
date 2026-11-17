"""
Performance Evaluation Script for Indian Sign Language (ISL) Recognition System
Implements Section 4.6 Performance Evaluation / 4.6.1 Evaluation Criteria:
1. Accuracy: Fraction of test samples classified correctly
2. Precision: Correctness of predictions made for each letter
3. Recall: Fraction of each letter's samples that are found
4. F1-score: Balance between precision and recall
5. Confusion matrix: Identifies commonly confused letter pairs
6. Per-frame latency: Time from frame to prediction (pipeline breakdown & FPS)
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Optional visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

# Ensure src is on sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from landmark_extractor import LandmarkExtractor

def load_data_and_model(data_path=None, model_path=None):
    # Locate data
    if data_path is None:
        candidates = ['data/landmarks.csv', '../data/landmarks.csv']
        for c in candidates:
            if os.path.exists(c):
                data_path = c
                break
    if not data_path or not os.path.exists(data_path):
        raise FileNotFoundError("Could not find 'data/landmarks.csv'. Please run data_preprocessing.py first.")

    # Locate model
    if model_path is None:
        candidates = ['models/rf_model.pkl', '../models/rf_model.pkl']
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                break
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError("Could not find 'models/rf_model.pkl'. Please run train_model.py first.")

    # Load dataset
    df = pd.read_csv(data_path)
    X = df.drop('label', axis=1).values
    y_raw = df['label'].astype(str).str.upper().values

    # Load model
    with open(model_path, 'rb') as f:
        saved_data = pickle.load(f)
        model = saved_data['model']
        le = saved_data['label_encoder']

    y_encoded = le.transform(y_raw)
    class_names = [str(c).upper() for c in le.classes_]

    return X, y_encoded, class_names, model, le

def evaluate_classification_metrics(X_test, y_test, class_names, model, output_dir='.'):
    """
    Computes Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
    """
    print("\n" + "="*70)
    print(" 4.6.1 EVALUATION METRICS REPORT")
    print("="*70)

    # Predictions
    y_pred = model.predict(X_test)

    # 1. Accuracy
    acc = accuracy_score(y_test, y_pred)
    
    # 2. Precision (macro & weighted)
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    prec_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # 3. Recall (macro & weighted)
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # 4. F1-score (macro & weighted)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Print Summary Table
    print(f"\n[+] OVERALL METRICS SUMMARY:")
    print("-" * 55)
    print(f" {'Metric':<25} | {'Score':<12} | {'Percentage':<10}")
    print("-" * 55)
    print(f" {'Accuracy':<25} | {acc:<12.4f} | {acc * 100:.2f}%")
    print(f" {'Precision (Macro)':<25} | {prec_macro:<12.4f} | {prec_macro * 100:.2f}%")
    print(f" {'Precision (Weighted)':<25} | {prec_weighted:<12.4f} | {prec_weighted * 100:.2f}%")
    print(f" {'Recall (Macro)':<25} | {rec_macro:<12.4f} | {rec_macro * 100:.2f}%")
    print(f" {'Recall (Weighted)':<25} | {rec_weighted:<12.4f} | {rec_weighted * 100:.2f}%")
    print(f" {'F1-score (Macro)':<25} | {f1_macro:<12.4f} | {f1_macro * 100:.2f}%")
    print(f" {'F1-score (Weighted)':<25} | {f1_weighted:<12.4f} | {f1_weighted * 100:.2f}%")
    print("-" * 55)

    # Per-Class Classification Report
    print("\n[+] PER-LETTER CLASSIFICATION METRICS (Precision, Recall, F1-Score):")
    print("-" * 70)
    report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    report_str = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    print(report_str)

    # 5. Confusion Matrix & Confused Letter Pairs
    cm = confusion_matrix(y_test, y_pred)
    confused_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i][j] > 0:
                confused_pairs.append({
                    'true_letter': class_names[i],
                    'predicted_as': class_names[j],
                    'count': int(cm[i][j])
                })

    confused_pairs.sort(key=lambda x: x['count'], reverse=True)

    print("\n[+] CONFUSION MATRIX ANALYSIS (Identified Confused Letter Pairs):")
    print("-" * 70)
    if confused_pairs:
        print(f" Found {len(confused_pairs)} confused sign occurrences on the test set:")
        for rank, p in enumerate(confused_pairs[:10], start=1):
            print(f"   {rank:2d}. Sign '{p['true_letter']}' misclassified as '{p['predicted_as']}' ({p['count']} test samples)")
    else:
        print("  [Perfect] Zero misclassifications across test set! Confusion matrix is completely diagonal.")

    # Save Confusion Matrix Heatmaps if matplotlib & seaborn are installed
    if HAS_PLOT:
        try:
            # 1. Raw Counts Confusion Matrix
            plt.figure(figsize=(14, 12))
            annot_counts = np.array([[f"{val}" if val > 0 else "" for val in row] for row in cm])
            sns.heatmap(cm, annot=annot_counts, fmt="", cmap='Blues',
                        linewidths=0.5, linecolor="#e0e0e0",
                        cbar_kws={'label': 'Number of Samples', 'shrink': 0.8},
                        xticklabels=class_names, yticklabels=class_names)
            plt.title(f'ISL Gesture Recognition - Confusion Matrix (Counts)\nOverall Test Accuracy: {acc*100:.2f}% | Test Samples: {len(y_test)}',
                      fontsize=14, fontweight='bold', pad=15)
            plt.xlabel('Predicted Letter (ISL)', fontsize=12, fontweight='bold', labelpad=10)
            plt.ylabel('True Letter (ISL)', fontsize=12, fontweight='bold', labelpad=10)
            plt.tight_layout()
            cm_path = os.path.join(output_dir, 'confusion_matrix.png')
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\n  [OK] Confusion matrix (Counts) saved to: {cm_path}")

            # 2. Normalized Percentage Confusion Matrix
            with np.errstate(divide='ignore', invalid='ignore'):
                cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                cm_norm = np.nan_to_num(cm_norm)
            plt.figure(figsize=(14, 12))
            annot_norm = np.array([[f"{val*100:.0f}%" if val > 0 else "" for val in row] for row in cm_norm])
            sns.heatmap(cm_norm * 100, annot=annot_norm, fmt="", cmap='Blues',
                        vmin=0, vmax=100, linewidths=0.5, linecolor="#e0e0e0",
                        cbar_kws={'label': 'Prediction Rate (%)', 'shrink': 0.8},
                        xticklabels=class_names, yticklabels=class_names)
            plt.title(f'ISL Gesture Recognition - Normalized Confusion Matrix (%)\nRecall per Class | Average F1: {f1_macro*100:.2f}%',
                      fontsize=14, fontweight='bold', pad=15)
            plt.xlabel('Predicted Letter (ISL)', fontsize=12, fontweight='bold', labelpad=10)
            plt.ylabel('True Letter (ISL)', fontsize=12, fontweight='bold', labelpad=10)
            plt.tight_layout()
            cm_norm_path = os.path.join(output_dir, 'confusion_matrix_normalized.png')
            plt.savefig(cm_norm_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  [OK] Normalized confusion matrix (%) saved to: {cm_norm_path}")
        except Exception as e:
            print(f"  [Notice] Could not render heatmap plot: {e}")

    return {
        'accuracy': acc,
        'precision_macro': prec_macro,
        'precision_weighted': prec_weighted,
        'recall_macro': rec_macro,
        'recall_weighted': rec_weighted,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confused_pairs': confused_pairs,
        'confusion_matrix': cm
    }

def benchmark_per_frame_latency(model, feature_dim=182, iterations=200):
    """
    Measures 6. Per-frame latency:
    - Time from raw video frame to prediction.
    - Benchmarks landmark extraction, geometric feature extraction, and model prediction.
    """
    print("\n" + "="*70)
    print(" 4.6.2 PER-FRAME LATENCY BENCHMARK (Time from Frame to Prediction)")
    print("="*70)
    print(f" Running {iterations} latency measurement iterations...")

    extractor = LandmarkExtractor(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.35, enable_smoothing=True)

    # Create dummy 640x480 frame simulating real webcam frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw simple simulated hand geometry
    import cv2
    cv2.circle(test_frame, (320, 240), 60, (200, 200, 200), -1)
    for offset_x in [-40, -20, 0, 20, 40]:
        cv2.line(test_frame, (320, 240), (320 + offset_x, 150), (220, 220, 220), 8)

    # 1. Warm-up
    dummy_input = np.zeros((1, feature_dim))
    for _ in range(10):
        _ = model.predict(dummy_input)
        _ = extractor.extract_landmarks(test_frame)

    # 2. Benchmark Landmark Extraction
    extraction_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = extractor.extract_landmarks(test_frame)
        t1 = time.perf_counter()
        extraction_times.append((t1 - t0) * 1000) # ms

    # 3. Benchmark Model Inference Latency
    inference_times = []
    sample_feature = np.random.randn(1, feature_dim).astype(np.float32)
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = model.predict(sample_feature)
        t1 = time.perf_counter()
        inference_times.append((t1 - t0) * 1000) # ms

    # Aggregates
    mean_ext = np.mean(extraction_times)
    std_ext = np.std(extraction_times)
    mean_inf = np.mean(inference_times)
    std_inf = np.std(inference_times)
    
    total_mean_latency = mean_ext + mean_inf
    fps = 1000.0 / total_mean_latency if total_mean_latency > 0 else 0

    print("\n[+] PIPELINE LATENCY BREAKDOWN:")
    print("-" * 65)
    print(f" {'Pipeline Component':<35} | {'Latency (ms)':<15} | {'Std Dev':<10}")
    print("-" * 65)
    print(f" {'1. MediaPipe Landmark Extraction':<35} | {mean_ext:>8.2f} ms     | ±{std_ext:.2f} ms")
    print(f" {'2. Random Forest Model Inference':<35} | {mean_inf:>8.2f} ms     | ±{std_inf:.2f} ms")
    print("-" * 65)
    print(f" {'Total Per-Frame Latency':<35} | {total_mean_latency:>8.2f} ms     |")
    print(f" {'Real-Time Throughput':<35} | {fps:>8.1f} FPS    |")
    print("-" * 65)

    return {
        'landmark_latency_ms': mean_ext,
        'inference_latency_ms': mean_inf,
        'total_latency_ms': total_mean_latency,
        'fps': fps
    }

def main():
    print("\n======================================================================")
    print("      INDIAN SIGN LANGUAGE SYSTEM - PERFORMANCE EVALUATION (4.6)      ")
    print("======================================================================")

    # 1. Load Data and Trained Model
    X, y, class_names, model, le = load_data_and_model()
    
    # 2. Holdout Test Split (matching training 80/20 stratified split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[*] Total Dataset: {len(X)} samples across {len(class_names)} classes")
    print(f"[*] Evaluation Holdout Test Set: {len(X_test)} samples (20% stratified test split)")

    # 3. Evaluate Metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)
    metrics = evaluate_classification_metrics(X_test, y_test, class_names, model)

    # 4. Measure Per-Frame Latency
    latency = benchmark_per_frame_latency(model, feature_dim=X.shape[1], iterations=100)

    # 5. Output Ready-to-Paste Summary for Project Report / Thesis Table
    print("\n" + "="*70)
    print(" 4.6.1 EVALUATION CRITERIA RESULTS SUMMARY TABLE (FOR REPORT/DOCS)")
    print("="*70)
    print(f"| {'Metric':<18} | {'Measured Value':<18} | {'Description / Purpose':<30} |")
    print(f"|{'-'*20}|{'-'*20}|{'-'*32}|")
    print(f"| {'Accuracy':<18} | {metrics['accuracy']*100:>6.2f}%            | Fraction of test samples classified correctly  |")
    print(f"| {'Precision':<18} | {metrics['precision_macro']*100:>6.2f}% (Macro)     | Correctness of predictions for each letter     |")
    print(f"| {'Recall':<18} | {metrics['recall_macro']*100:>6.2f}% (Macro)     | Fraction of each letter's samples found        |")
    print(f"| {'F1-score':<18} | {metrics['f1_macro']*100:>6.2f}% (Macro)     | Balance between precision and recall           |")
    print(f"| {'Confusion matrix':<18} | {len(metrics['confused_pairs']):>2d} confused pairs   | Identifies commonly confused letter pairs      |")
    print(f"| {'Per-frame latency':<18} | {latency['total_latency_ms']:>6.2f} ms ({latency['fps']:.1f} FPS) | Time from frame to prediction                  |")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
