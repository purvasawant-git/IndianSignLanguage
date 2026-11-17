"""
Indian Sign Language (ISL) Recognition System - Confusion Matrix Generator
Generates publication-quality Confusion Matrix plots (Count and Normalized %)
and detailed per-class classification metrics.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

def generate_confusion_matrices(
    data_path='data/landmarks.csv',
    model_path='models/rf_model.pkl',
    output_dir='.',
    dpi=300
):
    print("=" * 65)
    print(" [*] GENERATING ISL RECOGNITION CONFUSION MATRIX")
    print("=" * 65)

    # 1. Check data and model existence
    if not os.path.exists(data_path):
        alt_data = os.path.join('..', data_path)
        if os.path.exists(alt_data):
            data_path = alt_data
        else:
            raise FileNotFoundError(f"Dataset not found at {data_path}")

    if not os.path.exists(model_path):
        alt_model = os.path.join('..', model_path)
        if os.path.exists(alt_model):
            model_path = alt_model
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")

    # 2. Load dataset
    print(f"[+] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    X = df.drop('label', axis=1).values
    y_raw = df['label'].astype(str).str.upper().values

    # 3. Load model and label encoder
    print(f"[+] Loading trained model from: {model_path}")
    with open(model_path, 'rb') as f:
        saved_dict = pickle.load(f)
    model = saved_dict['model']
    le = saved_dict['label_encoder']

    y_encoded = le.transform(y_raw)
    class_names = [str(c).upper() for c in le.classes_]
    n_classes = len(class_names)

    # 4. Stratified 80/20 train/test split matching model training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"[+] Dataset size: {len(X)} samples across {n_classes} classes")
    print(f"[+] Test set size: {len(X_test)} samples (20% holdout test split)")

    # 5. Model prediction & evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    print(f"\n[+] Model Test Accuracy : {accuracy * 100:.2f}%")
    print(f"[+] Macro Precision     : {prec * 100:.2f}%")
    print(f"[+] Macro Recall        : {rec * 100:.2f}%")
    print(f"[+] Macro F1-Score      : {f1 * 100:.2f}%")

    # 6. Calculate Confusion Matrix
    cm_counts = confusion_matrix(y_test, y_pred)
    # Normalized confusion matrix (recall per row)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm_counts.astype('float') / cm_counts.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    # Identify confused pairs
    confused_pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm_counts[i, j] > 0:
                confused_pairs.append({
                    'true': class_names[i],
                    'predicted': class_names[j],
                    'count': int(cm_counts[i, j]),
                    'error_rate': float(cm_norm[i, j] * 100)
                })
    confused_pairs.sort(key=lambda x: x['count'], reverse=True)

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------
    # PLOT 1: Confusion Matrix (Counts)
    # -------------------------------------------------------------
    print("\n[+] Generating Confusion Matrix Plot (Counts)...")
    plt.figure(figsize=(14, 12))
    sns.set_theme(style="white")
    
    # Custom annotations formatting (display number or empty if 0 for cleaner look)
    annot_counts = np.array([[f"{val}" if val > 0 else "" for val in row] for row in cm_counts])
    
    ax1 = sns.heatmap(
        cm_counts,
        annot=annot_counts,
        fmt="",
        cmap="Blues",
        linewidths=0.5,
        linecolor="#e0e0e0",
        cbar_kws={'label': 'Number of Samples', 'shrink': 0.8},
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 9, "fontweight": "normal"}
    )
    plt.title(
        f"ISL Gesture Recognition - Confusion Matrix (Counts)\nOverall Test Accuracy: {accuracy * 100:.2f}% | Test Samples: {len(y_test)}",
        fontsize=14,
        fontweight='bold',
        pad=16
    )
    plt.xlabel("Predicted Letter (ISL)", fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel("True Letter (ISL)", fontsize=12, fontweight='bold', labelpad=10)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    count_plot_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(count_plot_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved count confusion matrix: {os.path.abspath(count_plot_path)}")

    # -------------------------------------------------------------
    # PLOT 2: Confusion Matrix (Normalized Percentage %)
    # -------------------------------------------------------------
    print("[+] Generating Normalized Confusion Matrix Plot (Percentages)...")
    plt.figure(figsize=(14, 12))
    
    annot_norm = np.array([[f"{val * 100:.0f}%" if val > 0 else "" for val in row] for row in cm_norm])
    
    ax2 = sns.heatmap(
        cm_norm * 100,
        annot=annot_norm,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor="#e0e0e0",
        cbar_kws={'label': 'Prediction Rate (%)', 'shrink': 0.8},
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 8.5, "fontweight": "normal"}
    )
    plt.title(
        f"ISL Gesture Recognition - Normalized Confusion Matrix (%)\nRecall per Class | Average F1: {f1 * 100:.2f}%",
        fontsize=14,
        fontweight='bold',
        pad=16
    )
    plt.xlabel("Predicted Letter (ISL)", fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel("True Letter (ISL)", fontsize=12, fontweight='bold', labelpad=10)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()

    norm_plot_path = os.path.join(output_dir, 'confusion_matrix_normalized.png')
    plt.savefig(norm_plot_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved normalized confusion matrix: {os.path.abspath(norm_plot_path)}")

    # 7. Print Confused Pairs Summary
    print("\n" + "=" * 65)
    print(" [*] CONFUSED LETTER PAIRS SUMMARY")
    print("=" * 65)
    if confused_pairs:
        print(f"Total misclassifications on holdout test set: {len(confused_pairs)} types of confusion:")
        for idx, item in enumerate(confused_pairs[:10], 1):
            print(f"  {idx:2d}. Sign '{item['true']}' misclassified as '{item['predicted']}': {item['count']} sample(s) ({item['error_rate']:.1f}%)")
    else:
        print("  [Perfect] Zero misclassifications! All signs correctly recognized on the test set.")
    print("=" * 65)

    # 8. Full Classification Report
    print("\n[+] Detailed Classification Report (Per-Sign):")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

    return {
        'accuracy': accuracy,
        'f1_macro': f1,
        'count_plot': count_plot_path,
        'norm_plot': norm_plot_path,
        'confused_pairs': confused_pairs,
        'class_names': class_names,
        'test_samples': len(y_test)
    }

if __name__ == '__main__':
    generate_confusion_matrices()
