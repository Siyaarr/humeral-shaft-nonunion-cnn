import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from scipy import stats
import sys
import argparse

# Add src to pythonpath so we can import your project modules
sys.path.append(os.path.join(os.getcwd(), 'src'))
try:
    from src import config
    from src import data_loader
    from src import model as model_builder
except ImportError:
    print("❌ Error: Could not import src modules. Make sure you run this script from the 'AA - MasterChef' directory.")
    sys.exit(1)

def _compute_metrics_at_threshold(y_true, y_pred_probs, threshold):
    """
    Compute threshold-dependent metrics from probabilities.
    """
    preds = (y_pred_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

    accuracy = accuracy_score(y_true, preds)
    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Sensitivity (Recall)": recall,
        "Specificity": specificity,
        "F1 Score": f1,
    }


def calculate_bootstrap_ci(
    y_true,
    y_pred_probs,
    *,
    fixed_threshold,
    n_bootstraps=2000,
    ci=0.95,
):
    """
    Calculates 95% Confidence Intervals using bootstrapping.

    Note:
    - ROC AUC / PR AUC are threshold-free.
    - Accuracy/Precision/Recall/F1/Specificity are computed at `fixed_threshold`
      to match the paper's "optimized threshold" reporting.
    """
    print(f"\n🔄 Calculating 95% CIs using {n_bootstraps} bootstraps...")
    print(f"🎯 Using fixed threshold: {fixed_threshold:.8f}")
    
    # Store bootstrapped metrics
    scores = {
        "AUC ROC": [],
        "AUC PR": [],
        "Accuracy": [],
        "Sensitivity (Recall)": [],
        "Specificity": [],
        "Precision": [],
        "F1 Score": [],
    }
    
    rng = np.random.RandomState(42)
    indices = np.arange(len(y_true))
    
    for i in range(n_bootstraps):
        if i % 200 == 0:
            print(f"   Bootstrap {i}/{n_bootstraps}...", end='\r')
            
        # Sample with replacement
        pred_idx = rng.choice(indices, len(indices), replace=True)
        boot_y_true = y_true[pred_idx]
        boot_y_pred_probs = y_pred_probs[pred_idx]
        
        # Skip if only one class is present in the sample (can happen in small datasets)
        if len(np.unique(boot_y_true)) < 2:
            continue
            
        # 1. Threshold-free metrics
        try:
            auc_roc = roc_auc_score(boot_y_true, boot_y_pred_probs)
            scores["AUC ROC"].append(auc_roc)
        except ValueError:
            continue

        # PR AUC (Average Precision)
        try:
            auc_pr = average_precision_score(boot_y_true, boot_y_pred_probs)
            scores["AUC PR"].append(auc_pr)
        except ValueError:
            # If something odd happens, skip this bootstrap
            continue

        # 2. Threshold-dependent metrics at the fixed threshold
        fixed_metrics = _compute_metrics_at_threshold(boot_y_true, boot_y_pred_probs, fixed_threshold)
        for k, v in fixed_metrics.items():
            scores[k].append(v)

    print("\n✅ Calculation complete.\n")
    
    # Compute and print CIs
    alpha = (1 - ci) / 2
    print("-" * 80)
    print(f"{'METRIC':<25} | {'POINT':<8} | {'95% CI':<20}")
    print("-" * 80)

    # Point estimates on the full test set (matches what you should report)
    point_estimates = {
        "AUC ROC": roc_auc_score(y_true, y_pred_probs),
        "AUC PR": average_precision_score(y_true, y_pred_probs),
        **_compute_metrics_at_threshold(y_true, y_pred_probs, fixed_threshold),
    }
    
    for metric, values in scores.items():
        if not values:
            continue
        lower = np.percentile(values, alpha * 100)
        upper = np.percentile(values, (1 - alpha) * 100)
        point = point_estimates.get(metric, float("nan"))
        
        # Format for Lancet: 0·55 (0·45–0·65)
        point_str = f"{point:.3f}".replace('.', '·')
        lower_str = f"{lower:.3f}".replace('.', '·')
        upper_str = f"{upper:.3f}".replace('.', '·')
        
        print(f"{metric:<25} | {point_str:<8} | ({lower_str}–{upper_str})")

def main():
    parser = argparse.ArgumentParser(description="Compute bootstrap 95% CIs for final model test metrics.")
    parser.add_argument("--n_bootstraps", type=int, default=2000, help="Number of bootstrap resamples (default: 2000).")
    parser.add_argument("--ci", type=float, default=0.95, help="CI level (default: 0.95).")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed threshold to use for threshold-dependent metrics. If omitted, uses final_test_results.json optimal_threshold.",
    )
    args = parser.parse_args()

    # Paths
    base_dir = os.getcwd()
    final_model_dir = os.path.join(base_dir, 'results - FINAL', 'final_model')
    # Adjust this path if your dataset location is different
    test_dir = os.path.join(base_dir, '..', '..', 'Dataset', 'EP_Versio3_Removed_Turhat_Lapiammuttu_BadViistot_Proteesi_Levy', 'test')
    
    details_path = os.path.join(final_model_dir, 'final_model_details.json')
    test_results_path = os.path.join(final_model_dir, 'final_test_results.json')
    weights_path = os.path.join(final_model_dir, 'final_fine_tuned_model_weights.h5')
    
    if not os.path.exists(details_path):
        print(f"❌ Error: {details_path} not found.")
        return
        
    print("🚀 Loading model details...")
    with open(details_path, 'r') as f:
        details = json.load(f)
        
    model_name = details['model_name']
    hyperparameters = details['hyperparameters']
    
    # Build model
    metadata = config.MODEL_METADATA[model_name]
    image_size = metadata.get('required_image_size', config.DEFAULT_IMAGE_SIZE)
    input_shape = (*image_size, 3)
    
    model = model_builder.build_model(
        base_model_loader=metadata['loader'],
        input_shape=input_shape,
        dense_units=config.FIXED_DENSE_UNITS,
        dropout_rate=hyperparameters['dropout_rate'],
        kernel_regularizer_l2=hyperparameters.get('kernel_regularizer_l2', 0.0),
        use_dense_layer=hyperparameters['use_dense_layer']
    )
    
    # Helper to load weights
    model_builder.unfreeze_layers(model)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    
    print(f"📥 Loading weights from {weights_path}...")
    model.load_weights(weights_path)
    
    # Prepare test data
    print("📂 Loading test dataset...")
    use_canny = hyperparameters.get('use_canny', False)
    preprocess_function = metadata['preprocess']
    
    test_ds, _, _ = data_loader.create_dataset(
        test_dir, image_size, config.BATCH_SIZE,
        preprocess_function, is_training=False, use_canny=use_canny, shuffle=False
    )
    
    # Get predictions
    print("🔮 Generating predictions...")
    y_true_list = []
    y_pred_probs_list = []
    
    for images, labels in test_ds:
        preds = model.predict_on_batch(images)
        y_pred_probs_list.append(preds.flatten())
        y_true_list.append(labels.numpy().flatten())
    
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred_probs = np.concatenate(y_pred_probs_list, axis=0)
    
    print(f"   Processed {len(y_true)} test images.")
    
    # Calculate CIs
    fixed_threshold = None
    if args.threshold is not None:
        fixed_threshold = float(args.threshold)
    else:
        if os.path.exists(test_results_path):
            with open(test_results_path, "r") as f:
                test_results = json.load(f)
            # Stored as string in your JSON
            fixed_threshold = float(test_results["final_test_metrics"]["optimal_threshold"])
        else:
            fixed_threshold = 0.5
            print(f"⚠️ Warning: {test_results_path} not found. Falling back to threshold=0.5")

    calculate_bootstrap_ci(
        y_true,
        y_pred_probs,
        fixed_threshold=fixed_threshold,
        n_bootstraps=args.n_bootstraps,
        ci=args.ci,
    )

if __name__ == "__main__":
    main()