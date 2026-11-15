import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import traceback
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, 
    precision_score, recall_score, f1_score, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from . import config
from . import data_loader
from . import model as model_builder

def evaluate_model_comprehensive(model_path, model_dir_name, hyperparameters, validation_data=None):
    """
    Performs comprehensive evaluation of a trained model including all metrics
    and generates detailed visualizations.
    
    Args:
        model_path (str): Path to the saved model's weights.
        model_dir_name (str): Name of the model directory for labeling.
        hyperparameters (dict): Dictionary of hyperparameters for rebuilding the model.
        validation_data: Validation dataset (optional, if None will load from config)
    
    Returns:
        dict: Dictionary containing all computed metrics
    """
    print(f"\n--- Comprehensive Evaluation for {model_dir_name} ---")
    
    # Load model by rebuilding it and loading weights
    try:
        # Extract base model name for metadata lookup
        if " - Dense_" in model_dir_name and " - Canny_" in model_dir_name:
            base_model_name = model_dir_name.split(" - Dense_")[0]
        else:
            base_model_name = model_dir_name
        
        metadata = config.MODEL_METADATA[base_model_name]
        image_size = metadata.get('required_image_size', config.DEFAULT_IMAGE_SIZE)
        input_shape = (*image_size, 3)

        # Re-build the model architecture
        print("Re-building model architecture for evaluation...")
        model = model_builder.build_model(
            base_model_loader=metadata['loader'],
            input_shape=input_shape,
            dense_units=config.FIXED_DENSE_UNITS,
            dropout_rate=hyperparameters['dropout_rate'],
            kernel_regularizer_l2=hyperparameters.get('kernel_regularizer_l2', 0.0),
            use_dense_layer=hyperparameters['use_dense_layer']
        )
        
        # Unfreeze layers and compile to match the state during training, as done in final_testing.py
        model_builder.unfreeze_layers(model)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Dummy optimizer for loading
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.AUC(name='auc')]
        )
        
        # Load the saved weights
        model.load_weights(model_path)
        print(f"✅ Model architecture re-built and weights loaded successfully from {model_path}")
    except Exception as e:
        print(f"❌ Error loading model for evaluation: {e}")
        traceback.print_exc()
        return None
    
    # Load validation data if not provided
    if validation_data is None:
        try:
            preprocess_function = metadata['preprocess']
            
            validation_data, _, _ = data_loader.create_dataset(
                config.VALIDATION_DIR, image_size, config.BATCH_SIZE, 
                preprocess_function, is_training=False
            )
            print(f"✅ Validation data loaded successfully")
        except Exception as e:
            print(f"❌ Error loading validation data: {e}")
            return None
    
    # Generate predictions
    print("Generating predictions...")
    y_true_list = []
    y_pred_probs_list = []
    
    try:
        for images, labels in validation_data:
            preds = model.predict_on_batch(images)
            y_pred_probs_list.append(preds.flatten())
            y_true_list.append(labels.numpy().flatten())
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return None
    
    if not y_true_list:
        print("❌ No validation data found")
        return None
    
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred_probs = np.concatenate(y_pred_probs_list, axis=0)
    
    # Calculate metrics at different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    results = {}
    
    print("\n--- Metrics at Different Thresholds ---")
    for threshold in thresholds:
        y_pred_classes = (y_pred_probs > threshold).astype(int)
        
        metrics = {
            'threshold': threshold,
            'accuracy': accuracy_score(y_true, y_pred_classes),
            'precision': precision_score(y_true, y_pred_classes, zero_division=0),
            'recall': recall_score(y_true, y_pred_classes, zero_division=0),
            'f1_score': f1_score(y_true, y_pred_classes, zero_division=0)
        }
        
        results[f'threshold_{threshold}'] = metrics
        print(f"Threshold {threshold}: Acc={metrics['accuracy']:.4f}, "
              f"Prec={metrics['precision']:.4f}, Rec={metrics['recall']:.4f}, "
              f"F1={metrics['f1_score']:.4f}")
    
    # Find optimal threshold for F1 score
    precisions, recalls, thresholds_pr = precision_recall_curve(y_true, y_pred_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    optimal_idx = np.argmax(f1_scores[:-1])  # Exclude last element
    optimal_threshold = thresholds_pr[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    y_pred_optimal = (y_pred_probs > optimal_threshold).astype(int)
    
    print(f"\n--- Optimal F1 Threshold: {optimal_threshold:.4f} (F1: {optimal_f1:.4f}) ---")
    
    # Calculate comprehensive metrics at optimal threshold
    optimal_metrics = {
        'optimal_threshold': optimal_threshold,
        'optimal_f1': optimal_f1,
        'accuracy': accuracy_score(y_true, y_pred_optimal),
        'precision': precision_score(y_true, y_pred_optimal, zero_division=0),
        'recall': recall_score(y_true, y_pred_optimal, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred_probs),
        'pr_auc': average_precision_score(y_true, y_pred_probs)
    }
    
    results['optimal'] = optimal_metrics
    
    print(f"Optimal Metrics: Acc={optimal_metrics['accuracy']:.4f}, "
          f"Prec={optimal_metrics['precision']:.4f}, Rec={optimal_metrics['recall']:.4f}, "
          f"ROC-AUC={optimal_metrics['roc_auc']:.4f}, PR-AUC={optimal_metrics['pr_auc']:.4f}")
    
    # Also calculate metrics at standard 0.5 threshold
    y_pred_standard = (y_pred_probs > 0.5).astype(int)
    standard_metrics = {
        'threshold': 0.5,
        'accuracy': accuracy_score(y_true, y_pred_standard),
        'precision': precision_score(y_true, y_pred_standard, zero_division=0),
        'recall': recall_score(y_true, y_pred_standard, zero_division=0),
        'f1_score': f1_score(y_true, y_pred_standard, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred_probs),
        'pr_auc': average_precision_score(y_true, y_pred_probs)
    }
    
    results['standard_0_5'] = standard_metrics
    
    print(f"\n--- Standard 0.5 Threshold Metrics ---")
    print(f"Standard Metrics: Acc={standard_metrics['accuracy']:.4f}, "
          f"Prec={standard_metrics['precision']:.4f}, Rec={standard_metrics['recall']:.4f}, "
          f"F1={standard_metrics['f1_score']:.4f}, ROC-AUC={standard_metrics['roc_auc']:.4f}, PR-AUC={standard_metrics['pr_auc']:.4f}")
    
    # Classification reports
    print(f"\n--- Classification Report (Optimal Threshold: {optimal_threshold:.4f}) ---")
    class_names = ['Healed', 'Non-healed']  # index 0 = healed (_0), index 1 = non-healed (_1)
    print(classification_report(y_true, y_pred_optimal, target_names=class_names, zero_division=0))
    
    print(f"\n--- Classification Report (Standard Threshold: 0.5) ---")
    print(classification_report(y_true, y_pred_standard, target_names=class_names, zero_division=0))
    
    # Store additional data for plotting
    results['evaluation_data'] = {
        'y_true': y_true,
        'y_pred_probs': y_pred_probs,
        'y_pred_optimal': y_pred_optimal,
        'y_pred_standard': y_pred_standard,
        'optimal_threshold': optimal_threshold
    }
    
    return results

def plot_comprehensive_evaluation(results, model_dir_name, output_dir):
    """
    Creates comprehensive evaluation plots including confusion matrix, ROC curve, etc.
    
    Args:
        results (dict): Results from evaluate_model_comprehensive
        model_dir_name (str): Name of the model directory
        output_dir (str): Directory to save plots
    """
    if not results or 'evaluation_data' not in results:
        print("❌ No evaluation data available for plotting")
        return
    
    eval_data = results['evaluation_data']
    y_true = eval_data['y_true']
    y_pred_probs = eval_data['y_pred_probs']
    y_pred_optimal = eval_data['y_pred_optimal']
    y_pred_standard = eval_data['y_pred_standard']
    optimal_threshold = eval_data['optimal_threshold']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    fig.suptitle(f'Comprehensive Evaluation - {model_dir_name}', fontsize=16)
    
    # 1. Confusion Matrix (Optimal Threshold)
    cm_optimal = confusion_matrix(y_true, y_pred_optimal)
    sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
    axes[0,0].set_title(f'Confusion Matrix\n(Optimal: {optimal_threshold:.3f})')
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('Actual')
    
    # 2. Confusion Matrix (Standard 0.5 Threshold)
    cm_standard = confusion_matrix(y_true, y_pred_standard)
    sns.heatmap(cm_standard, annot=True, fmt='d', cmap='Greens', ax=axes[0,1])
    axes[0,1].set_title('Confusion Matrix\n(Standard: 0.5)')
    axes[0,1].set_xlabel('Predicted')
    axes[0,1].set_ylabel('Actual')
    
    # 3. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = roc_auc_score(y_true, y_pred_probs)
    axes[0,2].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    axes[0,2].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0,2].set_xlim([0.0, 1.0])
    axes[0,2].set_ylim([0.0, 1.05])
    axes[0,2].set_xlabel('False Positive Rate')
    axes[0,2].set_ylabel('True Positive Rate')
    axes[0,2].set_title('ROC Curve')
    axes[0,2].legend(loc="lower right")
    axes[0,2].grid(True)
    
    # 4. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    pr_auc = average_precision_score(y_true, y_pred_probs)
    axes[0,3].plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    axes[0,3].set_xlabel('Recall')
    axes[0,3].set_ylabel('Precision')
    axes[0,3].set_title('Precision-Recall Curve')
    axes[0,3].legend(loc="lower left")
    axes[0,3].grid(True)
    
    # 5. Threshold vs Metrics
    thresholds = np.linspace(0.1, 0.9, 50)
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for thresh in thresholds:
        y_pred_thresh = (y_pred_probs > thresh).astype(int)
        accuracies.append(accuracy_score(y_true, y_pred_thresh))
        precisions.append(precision_score(y_true, y_pred_thresh, zero_division=0))
        recalls.append(recall_score(y_true, y_pred_thresh, zero_division=0))
        f1_scores.append(f1_score(y_true, y_pred_thresh, zero_division=0))
    
    axes[1,0].plot(thresholds, accuracies, label='Accuracy', linewidth=2)
    axes[1,0].plot(thresholds, precisions, label='Precision', linewidth=2)
    axes[1,0].plot(thresholds, recalls, label='Recall', linewidth=2)
    axes[1,0].plot(thresholds, f1_scores, label='F1 Score', linewidth=2)
    axes[1,0].axvline(x=optimal_threshold, color='red', linestyle='--', alpha=0.7, label=f'Optimal (F1): {optimal_threshold:.3f}')
    axes[1,0].axvline(x=0.5, color='green', linestyle='--', alpha=0.7, label='Standard: 0.5')
    axes[1,0].set_xlabel('Threshold')
    axes[1,0].set_ylabel('Score')
    axes[1,0].set_title('Metrics vs Threshold')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # 6. Prediction Distribution
    axes[1,1].hist(y_pred_probs[y_true == 0], bins=30, alpha=0.7, label='Healed', color='blue')
    axes[1,1].hist(y_pred_probs[y_true == 1], bins=30, alpha=0.7, label='Non-healed', color='red')
    axes[1,1].axvline(x=optimal_threshold, color='red', linestyle='--', alpha=0.7, label=f'Optimal: {optimal_threshold:.3f}')
    axes[1,1].axvline(x=0.5, color='green', linestyle='--', alpha=0.7, label='Standard: 0.5')
    axes[1,1].set_xlabel('Predicted Probability')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Prediction Distribution by Class')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    # 7. Metrics Comparison Table
    axes[1,2].axis('off')
    metrics_text = f"""
    Model: {model_dir_name}
    
    OPTIMAL THRESHOLD ({optimal_threshold:.4f}):
    Accuracy: {results['optimal']['accuracy']:.4f}
    Precision: {results['optimal']['precision']:.4f}
    Recall: {results['optimal']['recall']:.4f}
    F1 Score: {results['optimal']['optimal_f1']:.4f}
    
    STANDARD THRESHOLD (0.5):
    Accuracy: {results['standard_0_5']['accuracy']:.4f}
    Precision: {results['standard_0_5']['precision']:.4f}
    Recall: {results['standard_0_5']['recall']:.4f}
    F1 Score: {results['standard_0_5']['f1_score']:.4f}
    
    ROC AUC: {results['optimal']['roc_auc']:.4f}
    PR AUC: {results['optimal']['pr_auc']:.4f}
    """
    axes[1,2].text(0.1, 0.9, metrics_text, fontsize=10, verticalalignment='top', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    # 8. Confusion Matrix Comparison
    axes[1,3].axis('off')
    cm_text = f"""
    CONFUSION MATRICES
    
    Optimal Threshold ({optimal_threshold:.3f}):
    TN: {cm_optimal[0,0]}, FP: {cm_optimal[0,1]}
    FN: {cm_optimal[1,0]}, TP: {cm_optimal[1,1]}
    
    Standard Threshold (0.5):
    TN: {cm_standard[0,0]}, FP: {cm_standard[0,1]}
    FN: {cm_standard[1,0]}, TP: {cm_standard[1,1]}
    
    Legend:
    TN = True Negative
    FP = False Positive
    FN = False Negative
    TP = True Positive
    """
    axes[1,3].text(0.1, 0.9, cm_text, fontsize=10, verticalalignment='top', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f'{model_dir_name.replace(" ", "_")}_comprehensive_evaluation.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Comprehensive evaluation plot saved to: {plot_path}")
    plt.close()

def prepare_gradcam_analysis(model_dir_name, run_id):
    """
    Prepares for Grad-CAM analysis by checking model compatibility and 
    setting up necessary components.
    
    Args:
        model_dir_name (str): Name of the model directory (can include configuration)
        run_id (str): ID of the specific run
        
    Returns:
        dict: Information about Grad-CAM readiness and components
    """
    print(f"\n--- Grad-CAM Analysis Preparation for {model_dir_name} ({run_id}) ---")
    
    model_path = os.path.join('results', model_dir_name, run_id, config.MODEL_CHECKPOINT_FILENAME)
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return None
    
    # Load hyperparameters for this run to rebuild the model
    hp_path = os.path.join('results', model_dir_name, run_id, config.HYPERPARAMETERS_FILENAME)
    if not os.path.exists(hp_path):
        print(f"❌ Hyperparameters file not found at {hp_path}")
        return None
    with open(hp_path, 'r') as f:
        hyperparameters = json.load(f)

    try:
        # Re-build the model architecture
        if " - Dense_" in model_dir_name and " - Canny_" in model_dir_name:
            base_model_name = model_dir_name.split(" - Dense_")[0]
        else:
            base_model_name = model_dir_name
        metadata = config.MODEL_METADATA[base_model_name]
        image_size = metadata.get('required_image_size', config.DEFAULT_IMAGE_SIZE)
        input_shape = (*image_size, 3)

        print("Re-building model architecture for Grad-CAM...")
        model = model_builder.build_model(
            base_model_loader=metadata['loader'],
            input_shape=input_shape,
            dense_units=config.FIXED_DENSE_UNITS,
            dropout_rate=hyperparameters['dropout_rate'],
            kernel_regularizer_l2=hyperparameters.get('kernel_regularizer_l2', 0.0),
            use_dense_layer=hyperparameters['use_dense_layer']
        )
        # Load the saved weights
        model.load_weights(model_path)
        print(f"✅ Model for Grad-CAM re-built and weights loaded successfully")
        
        # Check if model is compatible with Grad-CAM
        base_model = model.layers[1]  # Base model is typically the second layer
        print(f"Base model: {base_model.name}")
        print(f"Base model type: {type(base_model)}")
        
        # Check for convolutional layers (required for Grad-CAM)
        conv_layers = []
        for i, layer in enumerate(base_model.layers):
            if 'conv' in layer.name.lower() or isinstance(layer, tf.keras.layers.Conv2D):
                conv_layers.append((i, layer.name, layer))
        
        if conv_layers:
            print(f"✅ Found {len(conv_layers)} convolutional layers - Grad-CAM compatible!")
            
            # Suggest good layers for Grad-CAM (typically the last few conv layers)
            suggested_layers = conv_layers[-3:] if len(conv_layers) >= 3 else conv_layers
            print(f"Suggested layers for Grad-CAM:")
            for idx, name, layer in suggested_layers:
                print(f"  - Layer {idx}: {name} (output shape: {layer.output_shape})")
            
            gradcam_info = {
                'compatible': True,
                'model_path': model_path,
                'base_model_name': base_model.name,
                'conv_layers': conv_layers,
                'suggested_layers': suggested_layers,
                'model_architecture': str(type(base_model))
            }
            
            print(f"✅ Grad-CAM preparation complete - ready for implementation!")
            return gradcam_info
        else:
            print(f"❌ No convolutional layers found - Grad-CAM not applicable")
            return None
            
    except Exception as e:
        print(f"❌ Error during Grad-CAM preparation: {e}")
        traceback.print_exc()
        return None

def analyze_results(model_dir_name):
    """
    Analyzes the results of all random search runs for a given model configuration.
    Enhanced version with comprehensive metrics and Grad-CAM preparation.
    
    This function works with ALL models defined in config.py automatically.

    Args:
        model_dir_name (str): The name of the model directory whose results are to be analyzed.
                             Can be either base model name (e.g., "MobileNetV3Large") or 
                             full directory name (e.g., "MobileNetV3Large - Dense_OFF - Canny_OFF")
    """
    # Extract base model name from directory name if it contains configuration info
    if " - Dense_" in model_dir_name and " - Canny_" in model_dir_name:
        # Extract base model name from full directory name
        base_model_name = model_dir_name.split(" - Dense_")[0]
    else:
        # Assume it's already a base model name
        base_model_name = model_dir_name
    
    # Validate base model name against config
    if base_model_name not in config.MODEL_METADATA:
        print(f"❌ Error: Base model '{base_model_name}' not found in config!")
        print("\n📋 Available models in config:")
        for i, name in enumerate(config.MODEL_METADATA.keys(), 1):
            print(f"  {i:2d}. {name}")
        print(f"\n💡 To analyze a specific model, use:")
        print(f"   python -m src.results_analyzer --model <MODEL_NAME> --dense_off --canny_off")
        print(f"\n💡 To see all available options, use:")
        print(f"   python -m src.analyze_all_models --list")
        return
    
    # Get model metadata for context using base model name
    metadata = config.MODEL_METADATA[base_model_name]
    input_size = metadata.get('required_image_size', config.DEFAULT_IMAGE_SIZE)
    
    print(f"\n🔍 Analyzing model: {base_model_name}")
    print(f"   Configuration: {model_dir_name}")
    print(f"   Architecture: {metadata['loader'].__name__}")
    print(f"   Input Size: {input_size}")
    print(f"   Preprocessing: {metadata['preprocess'].__name__}")
    
    model_results_dir = os.path.join('results', model_dir_name)
    if not os.path.isdir(model_results_dir):
        print(f"\n❌ Results directory for model '{model_dir_name}' not found!")
        print(f"   Expected location: {os.path.abspath(model_results_dir)}")
        print(f"\n💡 To train this model, run:")
        print(f"   python -m src.train --model {base_model_name}")
        print(f"\n💡 To check which models are trained, run:")
        print(f"   python -m src.analyze_all_models --check")
        return

    run_dirs = [d for d in os.listdir(model_results_dir)
                if os.path.isdir(os.path.join(model_results_dir, d)) and d.startswith('run_')]
    
    if not run_dirs:
        print(f"No run directories found for model '{model_dir_name}'.")
        return

    all_runs_data = []

    print(f"\n--- Analyzing Results for Model: {model_dir_name} ---")

    for run_dir_name in run_dirs:
        run_path = os.path.join(model_results_dir, run_dir_name)
        history_path = os.path.join(run_path, 'history.json')

        if not os.path.exists(history_path):
            print(f"Skipping {run_dir_name}: missing history.json")
            continue

        try:
            with open(history_path, 'r') as f:
                run_summary = json.load(f)
            
            # Adapt to the new history file structure from train.py
            history = run_summary.get('keras_history', {})
            hps = run_summary.get('hyperparameters', {})
            training_summary = run_summary.get('training_summary', {})
            
            if not history or not hps:
                print(f"Skipping {run_dir_name}: history.json is malformed or missing critical data.")
                continue

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Skipping {run_dir_name}: failed to read or parse history.json. Error: {e}")
            continue

        # Get multiple metrics from history and summary
        val_auc = history.get('val_auc', [0])
        val_loss = history.get('val_loss', [1000])
        train_auc = history.get('auc', [0])
        
        # Use pre-calculated best values from training summary if available
        best_val_auc = training_summary.get('best_val_auc', max(val_auc) if val_auc else 0)
        best_epoch = training_summary.get('best_epoch', np.argmax(val_auc) + 1 if val_auc else 0)
        
        final_val_loss = val_loss[-1] if val_loss else 1000
        final_train_auc = train_auc[-1] if train_auc else 0

        run_data = {
            'run_id': run_dir_name,
            'best_val_auc': best_val_auc,
            'best_epoch': best_epoch,
            'final_val_loss': final_val_loss,
            'final_train_auc': final_train_auc,
            'overfitting_score': abs(final_train_auc - best_val_auc),  # Simple overfitting indicator
            **hps  # Add all hyperparameters to the dictionary
        }
        all_runs_data.append(run_data)

    if not all_runs_data:
        print("No valid run data found to analyze.")
        return

    # Create a DataFrame for easy sorting and analysis
    results_df = pd.DataFrame(all_runs_data)
    results_df = results_df.sort_values(by='best_val_auc', ascending=False)
    
    # Identify the best run
    best_run = results_df.iloc[0]

    # --- Generate Enhanced Summary Report ---
    summary_path = os.path.join(model_results_dir, 'enhanced_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"=== ENHANCED ANALYSIS SUMMARY FOR {model_dir_name} ===\n\n")
        f.write(f"Total runs analyzed: {len(results_df)}\n")
        f.write(f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("=== BEST PERFORMING RUN ===\n")
        f.write(f"Run ID: {best_run['run_id']}\n")
        f.write(f"Best Validation AUC: {best_run['best_val_auc']:.4f} (at epoch {best_run['best_epoch']})\n")
        f.write(f"Final Validation Loss: {best_run['final_val_loss']:.4f}\n")
        f.write(f"Final Train AUC: {best_run['final_train_auc']:.4f}\n")
        f.write(f"Overfitting Score: {best_run['overfitting_score']:.4f}\n\n")
        
        f.write("Best Hyperparameters:\n")
        best_hps = best_run.drop(['run_id', 'best_val_auc', 'best_epoch', 'final_val_loss', 'final_train_auc', 'overfitting_score'])
        for key, value in best_hps.items():
            f.write(f"  - {key}: {value}\n")
            
        f.write("\n=== PERFORMANCE STATISTICS ===\n")
        if len(results_df) > 1:
            f.write(f"Average Validation AUC: {results_df['best_val_auc'].mean():.4f} ± {results_df['best_val_auc'].std():.4f}\n")
            f.write(f"Best Validation AUC: {results_df['best_val_auc'].max():.4f}\n")
            f.write(f"Worst Validation AUC: {results_df['best_val_auc'].min():.4f}\n")
            f.write(f"Average Overfitting Score: {results_df['overfitting_score'].mean():.4f} ± {results_df['overfitting_score'].std():.4f}\n\n")
        else:
            f.write(f"Single run analysis:\n")
            f.write(f"Validation AUC: {results_df['best_val_auc'].iloc[0]:.4f}\n")
            f.write(f"Overfitting Score: {results_df['overfitting_score'].iloc[0]:.4f}\n\n")
        
        f.write("=== ALL RUNS SUMMARY ===\n")
        f.write(results_df.to_string(index=False))

    print(f"Enhanced analysis complete. Summary report saved to: {summary_path}")

    # --- Perform Comprehensive Evaluation on Best Model ---
    best_model_path = os.path.join(model_results_dir, best_run['run_id'], config.MODEL_CHECKPOINT_FILENAME)

    # Load hyperparameters for the best run to rebuild the model
    best_hps = {}
    best_hp_path = os.path.join(model_results_dir, best_run['run_id'], config.HYPERPARAMETERS_FILENAME)
    if os.path.exists(best_hp_path):
        with open(best_hp_path, 'r') as f:
            best_hps = json.load(f)
    else:
        print(f"⚠️  Warning: Hyperparameters file not found for best run at {best_hp_path}. Cannot perform evaluation.")

    if os.path.exists(best_model_path) and best_hps:
        print(f"\n--- Performing Comprehensive Evaluation on Best Model ---")
        comprehensive_results = evaluate_model_comprehensive(best_model_path, model_dir_name, best_hps)
        
        if comprehensive_results:
            # Generate comprehensive plots
            plot_comprehensive_evaluation(comprehensive_results, model_dir_name, model_results_dir)
            
            # Save comprehensive metrics
            metrics_path = os.path.join(model_results_dir, 'comprehensive_metrics.json')
            # Remove evaluation_data before saving (contains numpy arrays) and convert numpy types
            metrics_to_save = {}
            for k, v in comprehensive_results.items():
                if k != 'evaluation_data':
                    if isinstance(v, dict):
                        # Handle nested dictionaries
                        metrics_to_save[k] = {}
                        for sub_k, sub_v in v.items():
                            if hasattr(sub_v, 'item'):  # NumPy scalar
                                metrics_to_save[k][sub_k] = sub_v.item()
                            else:
                                metrics_to_save[k][sub_k] = float(sub_v) if isinstance(sub_v, (np.floating, np.integer)) else sub_v
                    else:
                        if hasattr(v, 'item'):  # NumPy scalar
                            metrics_to_save[k] = v.item()
                        else:
                            metrics_to_save[k] = float(v) if isinstance(v, (np.floating, np.integer)) else v
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics_to_save, f, indent=4)
            print(f"📊 Comprehensive metrics saved to: {metrics_path}")
    
    # --- Prepare Grad-CAM Analysis ---
    gradcam_info = prepare_gradcam_analysis(model_dir_name, best_run['run_id'])
    if gradcam_info:
        gradcam_path = os.path.join(model_results_dir, 'gradcam_info.json')
        # Convert non-serializable objects to strings
        gradcam_serializable = {}
        for key, value in gradcam_info.items():
            if key == 'conv_layers' or key == 'suggested_layers':
                gradcam_serializable[key] = [(idx, name, str(layer)) for idx, name, layer in value]
            else:
                gradcam_serializable[key] = str(value) if not isinstance(value, (str, bool, int, float)) else value
        
        with open(gradcam_path, 'w') as f:
            json.dump(gradcam_serializable, f, indent=4)
        print(f"🔍 Grad-CAM preparation info saved to: {gradcam_path}")

    # --- Generate Enhanced Performance Plot ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Best run training curves
    best_run_path = os.path.join(model_results_dir, best_run['run_id'])
    best_history_path = os.path.join(best_run_path, 'history.json')
    with open(best_history_path, 'r') as f:
        best_run_summary = json.load(f)
    best_history = best_run_summary.get('keras_history', {})

    epochs = range(1, len(best_history.get('val_auc', [])) + 1)
    if not epochs:
        print("⚠️  Warning: Could not plot training curves, history is empty.")
    else:
        ax1.plot(epochs, best_history['val_auc'], label="Validation AUC", color='crimson', linewidth=2)
        ax1.plot(epochs, best_history['auc'], label="Training AUC", color='dodgerblue', linestyle='--', linewidth=2)
        ax1.scatter(best_run['best_epoch'], best_run['best_val_auc'],
                   marker='*', s=200, color='gold', zorder=5, label=f'Best Epoch ({best_run["best_epoch"]})')
        ax1.set_title(f'Training Curves - Best Run ({best_run["run_id"]})')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('AUC')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Loss curves
        ax2.plot(epochs, best_history['val_loss'], label="Validation Loss", color='red', linewidth=2)
        ax2.plot(epochs, best_history['loss'], label="Training Loss", color='blue', linestyle='--', linewidth=2)
        ax2.set_title('Loss Curves - Best Run')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
    
    # Plot 3: Hyperparameter analysis
    if len(results_df) > 1 and 'learning_rate_fine_tune' in results_df.columns and 'dropout_rate' in results_df.columns:
        scatter = ax3.scatter(results_df['learning_rate_fine_tune'], results_df['best_val_auc'], 
                   alpha=0.7, s=60, c=results_df['dropout_rate'], cmap='viridis')
        ax3.set_xlabel('Learning Rate (Fine-tune)')
        ax3.set_ylabel('Best Validation AUC')
        ax3.set_title('Hyperparameter Analysis')
        ax3.set_xscale('log')
        cbar = fig.colorbar(scatter, ax=ax3)
        cbar.set_label('Dropout Rate')
        ax3.grid(True)
    else:
        ax3.text(0.5, 0.5, 'Need >1 run with varied\nhyperparameters for analysis', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Hyperparameter Analysis')
    
    # Plot 4: Run comparison
    run_names = [f"Run {i+1}" for i in range(len(results_df))]
    bars = ax4.bar(run_names, results_df['best_val_auc'], 
                   color=['gold' if i == 0 else 'skyblue' for i in range(len(results_df))])
    ax4.set_title('Validation AUC Comparison Across Runs')
    ax4.set_xlabel('Runs (Sorted by Performance)')
    ax4.set_ylabel('Best Validation AUC')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, results_df['best_val_auc']):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(f'Enhanced Performance Analysis - {model_dir_name}', fontsize=16)
    plt.tight_layout()
    
    plot_path = os.path.join(model_results_dir, 'enhanced_performance_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Enhanced performance plot saved to: {plot_path}")
    plt.close()

    print(f"\n🎉 Enhanced analysis complete for {model_dir_name}!")
    print(f"📁 Check the results directory: {model_results_dir}")
    if gradcam_info and gradcam_info['compatible']:
        print(f"🔍 Model is ready for Grad-CAM analysis!")
        print(f"   Suggested layers: {[layer[1] for layer in gradcam_info['suggested_layers']]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Enhanced results analyzer with comprehensive metrics and Grad-CAM preparation.")
    parser.add_argument(
        '--model',
        type=str,
        help="The name of the model to analyze."
    )
    parser.add_argument(
        '--compare_all',
        action='store_true',
        help="Compare all trained models and show winner."
    )
    parser.add_argument(
        '--dense_on',
        action='store_true',
        help="Analyze Dense_ON variant of the model."
    )
    parser.add_argument(
        '--dense_off',
        action='store_true',
        help="Analyze Dense_OFF variant of the model."
    )
    parser.add_argument(
        '--canny_on',
        action='store_true',
        help="Analyze Canny_ON variant of the model."
    )
    parser.add_argument(
        '--canny_off',
        action='store_true',
        help="Analyze Canny_OFF variant of the model."
    )
    
    # Example usage:
    # python -m src.results_analyzer --model VGG16 --dense_off --canny_off
    # python -m src.results_analyzer --model MobileNetV3Large --dense_on --canny_on
    # python -m src.results_analyzer --model ResNet152 --dense_off --canny_on
    # python -m src.results_analyzer --compare_all
    
    parsed_args = parser.parse_args()
    
    if parsed_args.compare_all:
        # Compare all models functionality, treating each subdirectory in results as a unique variant
        print("🔍 COMPARING ALL TRAINED MODELS AND VARIANTS")
        print("="*60)
        
        if not os.path.exists(config.RESULTS_BASE_DIR):
            print(f"❌ Results directory not found: {config.RESULTS_BASE_DIR}")
            exit(1)
        
        all_models_data = []
        
        # Iterate over all subdirectories in the results folder.
        # Each subdirectory is treated as a unique model/variant.
        model_variants = sorted([d for d in os.listdir(config.RESULTS_BASE_DIR) if os.path.isdir(os.path.join(config.RESULTS_BASE_DIR, d))])

        for model_name in model_variants:
            model_dir = os.path.join(config.RESULTS_BASE_DIR, model_name)
            
            # Find the best run within this model's directory
            run_dirs = [d for d in os.listdir(model_dir) if d.startswith('run_') and os.path.isdir(os.path.join(model_dir, d))]
            if not run_dirs:
                print(f"⚠️  {model_name}: No run directories found.")
                continue

            best_auc_for_model = 0
            best_run_for_model = None
            
            for run_dir in run_dirs:
                history_path = os.path.join(model_dir, run_dir, 'history.json')
                if os.path.exists(history_path):
                    try:
                        with open(history_path, 'r') as f:
                            run_summary = json.load(f)
                        
                        # Check for new and old format for backward compatibility
                        if 'keras_history' in run_summary:
                            history = run_summary.get('keras_history', {})
                            training_summary = run_summary.get('training_summary', {})
                            val_auc = training_summary.get('best_val_auc', 0)
                        else: # Old format
                            history = run_summary
                            val_auc = max(history.get('val_auc', [0])) if history.get('val_auc') else 0

                        if val_auc > best_auc_for_model:
                            best_auc_for_model = val_auc
                            best_run_for_model = run_dir
                    except (json.JSONDecodeError, ValueError, KeyError):
                        print(f"⚠️  Could not parse history.json in {run_dir} for {model_name}. Skipping.")

            if best_run_for_model:
                all_models_data.append({
                    'Model/Variant': model_name,
                    'Best Validation AUC': best_auc_for_model,
                    'Best Run ID': best_run_for_model
                })
        
        if all_models_data:
            # Create a DataFrame, sort by AUC, and display results
            results_df = pd.DataFrame(all_models_data)
            results_df = results_df.sort_values(by='Best Validation AUC', ascending=False).reset_index(drop=True)
            
            print("\n--- Overall Model Comparison (Best Run per Model/Variant) ---")
            print(results_df.to_string())
            
            winner = results_df.iloc[0]
            print("\n" + "="*60)
            print(f"🏆 OVERALL WINNER: '{winner['Model/Variant']}'")
            print(f"   Best Validation AUC: {winner['Best Validation AUC']:.4f}")
            print(f"   In Run: {winner['Best Run ID']}")
            print("="*60)
        else:
            print("❌ No valid trained models found to compare!")
            
    elif parsed_args.model:
        # Validate conflicting options
        if parsed_args.dense_on and parsed_args.dense_off:
            print("❌ Error: Cannot specify both --dense_on and --dense_off")
            parser.print_help()
            exit(1)
        
        if parsed_args.canny_on and parsed_args.canny_off:
            print("❌ Error: Cannot specify both --canny_on and --canny_off")
            parser.print_help()
            exit(1)
        
        # Construct model directory name based on configuration flags
        model_name = parsed_args.model
        
        # Determine Dense configuration
        if parsed_args.dense_on:
            dense_str = "Dense_ON"
        elif parsed_args.dense_off:
            dense_str = "Dense_OFF"
        else:
            # Default to OFF if no dense option specified
            dense_str = "Dense_OFF"
        
        # Determine Canny configuration
        if parsed_args.canny_on:
            canny_str = "Canny_ON"
        elif parsed_args.canny_off:
            canny_str = "Canny_OFF"
        else:
            # Default to OFF if no canny option specified
            canny_str = "Canny_OFF"
        
        # Construct full model directory name
        model_dir_name = f"{model_name} - {dense_str} - {canny_str}"
        
        print(f"Analyzing model configuration: {model_dir_name}")
        analyze_results(model_dir_name)
    else:
        parser.print_help() 