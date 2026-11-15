#!/usr/bin/env python3
"""
Regenerate final-test figures with corrected class labels.

This script:
  1. Loads the saved final model weights (no retraining).
  2. Runs predictions on the held-out test set.
  3. Saves NEW figures into a fresh `corrected_plots/` subfolder, with:
        - Class 0 labeled "Healed"
        - Class 1 labeled "Non-healed"
  4. Does NOT modify or overwrite any existing files.

Class convention (verified from filesystem):
    *_0.jpg  ->  Healed       (majority, ~73.5% of train)
    *_1.jpg  ->  Non-healed   (minority, ~26.5% of train; clinical event of interest)

Usage (from the MasterChef folder, same as final_testing.py):
    python -m src.regenerate_plots
    python -m src.regenerate_plots --final-model-folder "results - FINAL/final_model"
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)

from . import config
from . import data_loader
from . import model as model_builder


CLASS_NAMES = ['Healed', 'Non-healed']  # index 0, index 1 — matches your file naming


def load_final_model(folder_path):
    details_path = os.path.join(folder_path, 'final_model_details.json')
    weights_path = os.path.join(folder_path, 'final_fine_tuned_model_weights.h5')
    if not os.path.exists(details_path):
        raise FileNotFoundError(f"Missing: {details_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing: {weights_path}")

    with open(details_path, 'r') as f:
        details = json.load(f)

    model_name = details['model_name']
    hp = details['hyperparameters']

    metadata = config.MODEL_METADATA[model_name]
    image_size = metadata.get('required_image_size', config.DEFAULT_IMAGE_SIZE)
    input_shape = (*image_size, 3)

    model = model_builder.build_model(
        base_model_loader=metadata['loader'],
        input_shape=input_shape,
        dense_units=config.FIXED_DENSE_UNITS,
        dropout_rate=hp['dropout_rate'],
        kernel_regularizer_l2=hp.get('kernel_regularizer_l2', 0.0),
        use_dense_layer=hp['use_dense_layer']
    )
    model_builder.unfreeze_layers(model)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )
    model.load_weights(weights_path)
    print(f"OK  Loaded {model_name} weights from {weights_path}")

    return model, model_name, hp, image_size, metadata['preprocess']


def predict_test_set(model, image_size, preprocess_function, use_canny):
    test_ds, n0, n1 = data_loader.create_dataset(
        config.TEST_DIR, image_size, config.BATCH_SIZE,
        preprocess_function, is_training=False, shuffle=False, use_canny=use_canny
    )
    print(f"OK  Test set: {n0+n1} samples (class 0 = Healed: {n0}, class 1 = Non-healed: {n1})")

    y_true_list, y_pred_list = [], []
    for images, labels in test_ds:
        preds = model.predict_on_batch(images).flatten()
        y_pred_list.append(preds)
        y_true_list.append(labels.numpy().flatten())
    y_true = np.concatenate(y_true_list).astype(int)
    y_pred_probs = np.concatenate(y_pred_list).astype(float)
    return y_true, y_pred_probs


def compute_optimal_threshold(y_true, y_pred_probs):
    precisions, recalls, thresholds_pr = precision_recall_curve(y_true, y_pred_probs)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    optimal_idx = int(np.argmax(f1s[:-1]))
    return float(thresholds_pr[optimal_idx])


def plot_combined(y_true, y_pred_probs, threshold, model_name, save_path):
    y_pred = (y_pred_probs > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Final Test Results - {model_name}', fontsize=14)

    for ax, lab in zip(axes.flatten(), ['(A)', '(B)', '(C)', '(D)']):
        ax.text(-0.1, 1.05, lab, transform=ax.transAxes,
                fontsize=16, va='top', ha='right')

    # (A) Confusion matrix with clinical class names
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES
    )
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')

    # (B) ROC curve  (positive class = Non-healed = class 1)
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = roc_auc_score(y_true, y_pred_probs)
    axes[0, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Baseline')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve (positive class: Non-healed)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # (C) Precision-Recall curve (positive class = Non-healed)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    pr_auc = average_precision_score(y_true, y_pred_probs)
    no_skill = float(np.sum(y_true == 1)) / len(y_true)
    axes[1, 0].plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.4f})')
    axes[1, 0].axhline(y=no_skill, color='r', linestyle='--',
                       label=f'Baseline ({no_skill:.2f})')
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision-Recall Curve (positive class: Non-healed)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # (D) Prediction distribution  — CORRECTED legend
    # Histogram shows P(class 1 = Non-healed) for each true class.
    axes[1, 1].hist(y_pred_probs[y_true == 0], bins=30, alpha=0.7,
                    label=CLASS_NAMES[0], color='blue')
    axes[1, 1].hist(y_pred_probs[y_true == 1], bins=30, alpha=0.7,
                    label=CLASS_NAMES[1], color='red')
    axes[1, 1].set_xlabel('Predicted Probability of Non-healed')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Prediction Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"OK  Saved {save_path}")


def plot_individual_confusion_matrix(y_true, y_pred_probs, threshold, model_name, save_path):
    cm = confusion_matrix(y_true, (y_pred_probs > threshold).astype(int))
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        cbar_kws={'label': 'Count'},
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES
    )
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"OK  Saved {save_path}")


def plot_individual_prediction_distribution(y_true, y_pred_probs, model_name, save_path):
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred_probs[y_true == 0], bins=30, alpha=0.7,
             label=f'{CLASS_NAMES[0]} (Class 0)', color='blue', edgecolor='black')
    plt.hist(y_pred_probs[y_true == 1], bins=30, alpha=0.7,
             label=f'{CLASS_NAMES[1]} (Class 1)', color='red', edgecolor='black')
    plt.xlabel('Predicted Probability of Non-healed')
    plt.ylabel('Frequency')
    plt.title(f'Prediction Distribution - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"OK  Saved {save_path}")


def print_metric_summary(y_true, y_pred_probs, threshold):
    y_pred = (y_pred_probs > threshold).astype(int)
    print("\n--- Metric sanity check (positive class = Non-healed = class 1) ---")
    print(f"  ROC AUC : {roc_auc_score(y_true, y_pred_probs):.4f}")
    print(f"  PR AUC  : {average_precision_score(y_true, y_pred_probs):.4f}")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  Recall   : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  F1       : {f1_score(y_true, y_pred, zero_division=0):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate final-test figures with corrected class labels (no retraining).'
    )
    parser.add_argument(
        '--final-model-folder', type=str,
        default=os.path.join('results - FINAL', 'final_model'),
        help='Folder containing final_model_details.json and final_fine_tuned_model_weights.h5'
    )
    parser.add_argument(
        '--output-subdir', type=str, default='corrected_plots',
        help='Subfolder name (inside --final-model-folder) for the new figures.'
    )
    args = parser.parse_args()

    folder = args.final_model_folder
    out_dir = os.path.join(folder, args.output_subdir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output folder: {out_dir}")

    model, model_name, hp, image_size, preprocess_function = load_final_model(folder)
    use_canny = hp.get('use_canny', False)

    y_true, y_pred_probs = predict_test_set(model, image_size, preprocess_function, use_canny)
    threshold = compute_optimal_threshold(y_true, y_pred_probs)
    print_metric_summary(y_true, y_pred_probs, threshold)

    plot_combined(
        y_true, y_pred_probs, threshold, model_name,
        os.path.join(out_dir, 'final_test_results.png')
    )
    plot_individual_confusion_matrix(
        y_true, y_pred_probs, threshold, model_name,
        os.path.join(out_dir, 'confusion_matrix.png')
    )
    plot_individual_prediction_distribution(
        y_true, y_pred_probs, model_name,
        os.path.join(out_dir, 'prediction_distribution.png')
    )

    # Also persist the raw arrays so plots can be re-styled later without re-running predictions.
    np.savez(
        os.path.join(out_dir, 'predictions.npz'),
        y_true=y_true, y_pred_probs=y_pred_probs, threshold=threshold
    )
    print(f"OK  Saved {os.path.join(out_dir, 'predictions.npz')}")

    print("\nDONE. Nothing existing was overwritten; all new files live in:")
    print(f"  {out_dir}")


if __name__ == '__main__':
    main()
