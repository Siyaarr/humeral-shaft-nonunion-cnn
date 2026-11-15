#!/usr/bin/env python3
"""
Final Testing Script - "Lopullinen testaus (Fine-tuning)"

This script implements the final phase of your systematic evaluation:
1. Finds the winning model architecture based on validation AUC
2. Combines train + validation data for final training
3. Fine-tunes the winning model with lower learning rate
4. Tests the final model on the held-out test set
5. Generates comprehensive final evaluation metrics

Usage:
    python -m src.final_testing                           # Automatic winner selection
    python -m src.final_testing --model VGG16            # Force specific model
    python -m src.final_testing --skip-training          # Only test existing model
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers, losses
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import traceback

from . import config
from . import data_loader
from . import model as model_builder
from . import results_analyzer

# Test directory configuration (separate from training config)
TEST_DIR = '../../Dataset/EP_Versio3_Removed_Turhat_Lapiammuttu_BadViistot_Proteesi_Levy/test'

# Final testing specific configuration
EPOCHS_FINAL_TRAINING = 11  # Reduced epochs for final training phase
EARLY_STOPPING_PATIENCE_FINAL = 8  # Reduced patience for faster convergence

# Set seeds for reproducibility
tf.random.set_seed(config.SEED)
np.random.seed(config.SEED)

def find_winning_model():
    """
    Analyzes all trained models and finds the winner based on validation AUC.
    
    Returns:
        tuple: (winning_model_name, best_val_auc, best_hyperparameters, best_run_id, best_model_path)
    """
    print("\n" + "="*80)
    print("🏆 FINDING THE WINNING MODEL ARCHITECTURE")
    print("="*80)
    
    if not os.path.exists(config.RESULTS_BASE_DIR):
        print(f"❌ Results directory not found: {config.RESULTS_BASE_DIR}")
        return None, None, None, None, None
    
    # Collect results from all models
    all_results = []
    
    # Iterate through each base model defined in the metadata
    for model_name in config.MODEL_METADATA.keys():
        
        # Check for all possible configurations
        for dense_state in ['ON', 'OFF']:
            for canny_state in ['ON', 'OFF']:
                model_dir_name = f"{model_name} - Dense_{dense_state} - Canny_{canny_state}"
                model_results_dir = os.path.join(config.RESULTS_BASE_DIR, model_dir_name)

                if not os.path.exists(model_results_dir):
                    continue  # This configuration was not trained, skip it
                
                print(f"🔎 Analyzing results for: {model_dir_name}")
                
                # Find run directories within the configuration folder
                run_dirs = [d for d in os.listdir(model_results_dir) 
                           if os.path.isdir(os.path.join(model_results_dir, d)) and d.startswith('run_')]
                
                if not run_dirs:
                    print(f"⚠️  {model_dir_name}: No valid runs found")
                    continue
                
                # Find the best run for this specific model configuration
                best_val_auc_config = 0
                best_run_data_config = None
                
                for run_dir in run_dirs:
                    history_path = os.path.join(model_results_dir, run_dir, config.HISTORY_FILENAME)
                    hp_path = os.path.join(model_results_dir, run_dir, config.HYPERPARAMETERS_FILENAME)
                    model_path = os.path.join(model_results_dir, run_dir, config.MODEL_CHECKPOINT_FILENAME)
                    
                    if all(os.path.exists(p) for p in [history_path, hp_path, model_path]):
                        with open(history_path, 'r') as f:
                            run_summary = json.load(f)
                        with open(hp_path, 'r') as f:
                            hyperparams = json.load(f)
                        
                        # Correctly parse the nested history file
                        keras_history = run_summary.get('keras_history', {})
                        val_auc = keras_history.get('val_auc', [0])
                        run_best_val_auc = max(val_auc) if val_auc else 0
                        
                        if run_best_val_auc > best_val_auc_config:
                            best_val_auc_config = run_best_val_auc
                            best_run_data_config = {
                                'model_name': model_name,  # Store the base model name
                                'model_dir_name': model_dir_name, # Store the full name for display
                                'run_id': run_dir,
                                'val_auc': run_best_val_auc,
                                'best_epoch': np.argmax(val_auc) + 1 if val_auc else 0,
                                'total_epochs': len(val_auc) if val_auc else 0,  # Total epochs actually run
                                'hyperparameters': hyperparams,
                                'model_path': model_path
                            }
                
                if best_run_data_config:
                    all_results.append(best_run_data_config)
                    print(f"✅ {model_dir_name}: Best Val AUC = {best_val_auc_config:.4f} ({best_run_data_config['run_id']})")

    if not all_results:
        print("❌ No trained models found!")
        return None, None, None, None, None
    
    # Find overall winner from all configurations
    winner = max(all_results, key=lambda x: x['val_auc'])
    
    print(f"\n🥇 WINNER: {winner['model_dir_name']}")
    print(f"   Best Validation AUC: {winner['val_auc']:.4f}")
    print(f"   Best Run: {winner['run_id']}")
    print(f"   Best Epoch: {winner['best_epoch']}")
    print(f"   Model Path: {winner['model_path']}")
    
    # Save winner summary
    winner_summary = {
        'winning_model': winner['model_name'],
        'winning_val_auc': winner['val_auc'],
        'winning_run': winner['run_id'],
        'winning_epoch': winner['best_epoch'],
        'winning_hyperparameters': winner['hyperparameters'],
        'model_path': winner['model_path'],
        'all_model_results': all_results
    }
    
    summary_path = os.path.join(config.RESULTS_BASE_DIR, 'winner_selection_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(winner_summary, f, indent=4, default=str)
    
    print(f"📊 Winner summary saved to: {summary_path}")
    
    return winner['model_name'], winner['val_auc'], winner['hyperparameters'], winner['run_id'], winner['model_path']

def find_specified_model_run(model_name, dense_state, canny_state):
    """
    Finds the best run for a given model configuration by directly analyzing
    its run directories, without relying on a pre-computed summary.
    Handles fully or partially specified configurations.
    
    Args:
        model_name (str): The base name of the model (e.g., 'VGG16').
        dense_state (str or None): 'ON' or 'OFF', or None to search both.
        canny_state (str or None): 'ON' or 'OFF', or None to search both.
        
    Returns:
        dict or None: The dictionary of the best matching run, or None if not found.
    """
    # Construct a search pattern based on provided arguments
    dense_str = f"Dense_{dense_state}" if dense_state else "Dense_*"
    canny_str = f"Canny_{canny_state}" if canny_state else "Canny_*"
    model_dir_pattern = f"{model_name} - {dense_str} - {canny_str}"
    
    search_path = os.path.join(config.RESULTS_BASE_DIR, model_dir_pattern)
    print(f"🔎 Searching for model directories matching pattern: '{search_path}'")
    
    matching_dirs = glob.glob(search_path)

    if not matching_dirs:
        print(f"❌ No results directory found matching pattern.")
        return None
        
    print(f"Found {len(matching_dirs)} matching configuration directory(s).")

    best_overall_run_data = None
    best_overall_val_auc = 0

    for model_results_dir in matching_dirs:
        model_dir_name_actual = os.path.basename(model_results_dir)
        print(f"--- Analyzing configuration: {model_dir_name_actual} ---")
        
        run_dirs = [d for d in os.listdir(model_results_dir)
                   if os.path.isdir(os.path.join(model_results_dir, d)) and d.startswith('run_')]
        
        if not run_dirs:
            print(f"  No valid runs found in {model_dir_name_actual}")
            continue

        for run_dir in run_dirs:
            history_path = os.path.join(model_results_dir, run_dir, config.HISTORY_FILENAME)
            hp_path = os.path.join(model_results_dir, run_dir, config.HYPERPARAMETERS_FILENAME)
            model_path = os.path.join(model_results_dir, run_dir, config.MODEL_CHECKPOINT_FILENAME)
            
            if all(os.path.exists(p) for p in [history_path, hp_path, model_path]):
                with open(history_path, 'r') as f:
                    run_summary = json.load(f)
                with open(hp_path, 'r') as f:
                    hyperparams = json.load(f)
                
                keras_history = run_summary.get('keras_history', {})
                val_auc = keras_history.get('val_auc', [0])
                run_best_val_auc = max(val_auc) if val_auc else 0
                
                if run_best_val_auc > best_overall_val_auc:
                    best_overall_val_auc = run_best_val_auc
                    best_overall_run_data = {
                        'model_name': model_name,
                        'model_dir_name': model_dir_name_actual,
                        'run_id': run_dir,
                        'val_auc': run_best_val_auc,
                        'best_epoch': np.argmax(val_auc) + 1 if val_auc else 0,
                        'total_epochs': len(val_auc) if val_auc else 0,  # Total epochs actually run
                        'hyperparameters': hyperparams,
                        'model_path': model_path
                    }
    
    if not best_overall_run_data:
        print(f"❌ No valid, complete runs found for model '{model_name}' matching the criteria.")
        return None
        
    return best_overall_run_data

def create_combined_dataset(model_name, hyperparameters):
    """
    Creates a combined training dataset from train + validation data.
    
    Args:
        model_name (str): Name of the winning model
        hyperparameters (dict): Hyperparameters of the winning model.
        
    Returns:
        tuple: (combined_dataset, total_samples_c0, total_samples_c1)
    """
    print(f"\n--- Creating Combined Train+Val Dataset for {model_name} ---")
    
    metadata = config.MODEL_METADATA[model_name]
    image_size = metadata.get('required_image_size', config.DEFAULT_IMAGE_SIZE)
    preprocess_function = metadata['preprocess']
    
    # Get the canny setting from the winning model's hyperparameters
    use_canny = hyperparameters.get('use_canny', False)
    if use_canny:
        print("Winning model used Canny edge detection. Applying to combined dataset.")
    
    print(f"Using image size: {image_size}")
    print(f"Using preprocessing: {preprocess_function.__name__}")
    
    # Load train dataset (WITHOUT oversampling for better real-world performance)
    print("Loading training data (without oversampling/augmentation for natural class distribution)...")
    train_ds, train_c0, train_c1 = data_loader.create_dataset(
        config.TRAIN_DIR, image_size, config.BATCH_SIZE, 
        preprocess_function, is_training=False, use_canny=use_canny
    )
    
    # Load validation dataset (WITHOUT oversampling for natural distribution)
    print("Loading validation data (without oversampling/augmentation for natural distribution)...")
    val_ds, val_c0, val_c1 = data_loader.create_dataset(
        config.VALIDATION_DIR, image_size, config.BATCH_SIZE, 
        preprocess_function, is_training=False, use_canny=use_canny
    )
    
    # Combine datasets (natural class distribution for realistic performance)
    print("Combining datasets (natural class distribution for realistic clinical performance)...")
    combined_ds = train_ds.concatenate(val_ds).shuffle(buffer_size=train_c0 + train_c1 + val_c0 + val_c1).repeat()
    
    total_c0 = train_c0 + val_c0
    total_c1 = train_c1 + val_c1
    total_samples = total_c0 + total_c1
    
    print(f"Combined dataset statistics:")
    print(f"  Total samples: {total_samples}")
    print(f"  Class 0 (Healed): {total_c0} ({total_c0/total_samples:.1%})")
    print(f"  Class 1 (Non-healed): {total_c1} ({total_c1/total_samples:.1%})")
    print(f"  Using NATURAL class distribution for realistic clinical performance")
    
    return combined_ds, total_c0, total_c1

def fine_tune_winning_model(model_name, hyperparameters, combined_dataset, total_c0, total_c1, total_epochs_run, winner_model_path, custom_epochs=None, output_dir=None):
    """
    Fine-tunes the winning model on combined train+val data.
    
    Args:
        model_name (str): Name of the winning model
        hyperparameters (dict): Best hyperparameters from random search
        combined_dataset: Combined train+val dataset
        total_c0 (int): Total samples in class 0.
        total_c1 (int): Total samples in class 1.
        total_epochs_run (int): The total number of epochs the winning model actually ran.
        winner_model_path (str): Path to the saved weights from the best run.
        custom_epochs (int, optional): Custom number of epochs to use instead of total_epochs_run.
        output_dir (str, optional): An optional output directory to save model-related files.
        
    Returns:
        tuple: (fine_tuned_model, training_history)
    """
    print(f"\n--- Fine-tuning {model_name} on Combined Dataset ---")
    
    metadata = config.MODEL_METADATA[model_name]
    image_size = metadata.get('required_image_size', config.DEFAULT_IMAGE_SIZE)
    input_shape = (*image_size, 3)
    
    # Build model with winning hyperparameters
    print("Building model architecture...")
    model = model_builder.build_model(
        base_model_loader=metadata['loader'],
        input_shape=input_shape,
        dense_units=config.FIXED_DENSE_UNITS,
        dropout_rate=hyperparameters['dropout_rate'],
        kernel_regularizer_l2=hyperparameters['kernel_regularizer_l2'],
        use_dense_layer=hyperparameters['use_dense_layer']
    )
    
    # Unfreeze layers BEFORE loading weights to match the model's state
    model_builder.unfreeze_layers(model)

    # Use lower learning rate for fine-tuning (10x lower than winning lr)
    fine_tune_lr = hyperparameters['learning_rate_fine_tune'] * 0.1
    print(f"Using fine-tuning learning rate: {fine_tune_lr}")
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=fine_tune_lr),
        loss=losses.BinaryCrossentropy(label_smoothing=config.LABEL_SMOOTHING_FACTOR),
        metrics=[tf.keras.metrics.AUC(name='auc'), 'accuracy']
    )
    
    
    # Load the weights from the best run
    if os.path.exists(winner_model_path):
        print(f"Loading winning model weights from: {winner_model_path}")
        model.load_weights(winner_model_path)
        print("Weights loaded successfully.")
    else:
        print(f"⚠️  Warning: Could not find winning model weights at '{winner_model_path}'.")
        print("The model will start from its pre-trained ImageNet weights for fine-tuning.")
    
    
    # Setup callbacks
    base_dir = output_dir if output_dir else config.RESULTS_BASE_DIR
    final_model_dir = os.path.join(base_dir, 'final_model')
    os.makedirs(final_model_dir, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(final_model_dir, 'final_fine_tuned_model_weights.h5'),
            monitor='loss',
            mode='min',
            save_best_only=False,
            save_weights_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='loss',
            mode='min',
            factor=hyperparameters['lr_scheduler_factor'],
            patience=hyperparameters['lr_scheduler_patience'],
            verbose=1,
            min_lr=config.LR_SCHEDULER_MIN_LR
        )
    ]
    
    # Calculate steps per epoch based on actual combined dataset size
    # Using natural class distribution (no oversampling) for realistic clinical performance
    total_samples = total_c0 + total_c1
    steps_per_epoch = max(1, total_samples // config.BATCH_SIZE)
    
    # No class balancing techniques - using natural distribution for better real-world performance
    class_weights = None
    print("Using NATURAL class distribution (no oversampling/class weighting) to avoid minority class bias.")
    
    # Use custom epochs if provided, otherwise use total epochs actually run
    final_epochs = custom_epochs if custom_epochs is not None else total_epochs_run
    epochs_source = "custom" if custom_epochs is not None else f"total epochs from original training ({total_epochs_run})"
        
    print(f"Starting fine-tuning...")
    print(f"  Epochs: {final_epochs} ({epochs_source})")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Learning rate: {fine_tune_lr}")
    
    # Fine-tune the model
    history = model.fit(
        combined_dataset,
        epochs=final_epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model details for reproducible loading
    final_model_details = {
        'model_name': model_name,
        'hyperparameters': hyperparameters
    }
    with open(os.path.join(final_model_dir, 'final_model_details.json'), 'w') as f:
        json.dump(final_model_details, f, indent=4)
    
    # Save training history
    history_dict = {}
    for key, values in history.history.items():
        history_dict[key] = [float(val) for val in values]
    
    with open(os.path.join(final_model_dir, 'final_training_history.json'), 'w') as f:
        json.dump(history_dict, f, indent=4)
    
    # Generate training history plots
    print("📊 Generating retraining history plots...")
    
    # Plot training metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Final Training History - {model_name}', fontsize=16)
    
    epochs_range = range(1, len(history.history['loss']) + 1)
    
    # Loss plot
    axes[0,0].plot(epochs_range, history.history['loss'], 'b-', label='Training Loss')
    axes[0,0].set_title('Training Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # AUC plot
    if 'auc' in history.history:
        axes[0,1].plot(epochs_range, history.history['auc'], 'g-', label='Training AUC')
        axes[0,1].set_title('Training AUC')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('AUC')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
    
    # Accuracy plot
    if 'accuracy' in history.history:
        axes[1,0].plot(epochs_range, history.history['accuracy'], 'r-', label='Training Accuracy')
        axes[1,0].set_title('Training Accuracy')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Accuracy')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
    
    # Learning Rate plot (if available)
    if 'lr' in history.history:
        axes[1,1].plot(epochs_range, history.history['lr'], 'm-', label='Learning Rate')
        axes[1,1].set_title('Learning Rate')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Learning Rate')
        axes[1,1].set_yscale('log')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    else:
        # If no LR history, show a summary text
        axes[1,1].text(0.5, 0.5, f'Final Training Summary:\n\n'
                       f'• Total Epochs: {len(epochs_range)}\n'
                       f'• Final Loss: {history.history["loss"][-1]:.4f}\n'
                       f'• Learning Rate: {fine_tune_lr:.2e}\n'
                       f'• Total Samples: {total_samples}\n'
                       f'• Class Distribution: Natural',
                       horizontalalignment='center', verticalalignment='center',
                       transform=axes[1,1].transAxes, fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        axes[1,1].set_title('Training Summary')
        axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(final_model_dir, 'final_training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training history plots saved to: {final_model_dir}/final_training_history.png")
    print("✅ Fine-tuning complete!")
    
    return model, history

def save_preprocessed_examples(test_ds, model, output_dir, model_name):
    """Saves a few examples of preprocessed test images for verification."""
    print("\n💾 Saving preprocessed examples for verification...")
    
    examples_dir = os.path.join(output_dir, 'preprocessed_examples')
    os.makedirs(examples_dir, exist_ok=True)
    
    try:
        image_batch, label_batch = next(iter(test_ds))
        pred_batch = model.predict_on_batch(image_batch).flatten()
        
        num_examples_to_save = min(8, image_batch.shape[0])

        for i in range(num_examples_to_save):
            image = image_batch[i]
            true_label = int(label_batch[i])
            pred_prob = pred_batch[i]
            
            # Re-scale image to 0-255 for visual inspection
            img_to_save = image.numpy()
            img_to_save = (img_to_save - img_to_save.min()) / (img_to_save.max() - img_to_save.min() + 1e-8)
            img_to_save = (img_to_save * 255.0).astype(np.uint8)

            filename = f"example_{i}_true_{true_label}_pred_{pred_prob:.2f}.png"
            filepath = os.path.join(examples_dir, filename)
            
            tf.keras.utils.save_img(filepath, img_to_save)
            
        print(f"✅ Saved {num_examples_to_save} examples to: {examples_dir}")
    except Exception as e:
        print(f"⚠️ Could not save preprocessed examples. Error: {e}")

def evaluate_final_model(model, base_model_name, hyperparameters):
    """
    Performs the final evaluation on the test set.

    Args:
        model (tf.keras.Model): The trained and fine-tuned model.
        base_model_name (str): The name of the base model architecture.
        hyperparameters (dict): The dictionary of hyperparameters for this run.

    Returns:
        dict: A dictionary containing evaluation metrics and data for plotting.
    """
    # 1. Prepare Test Data
    print("\n--- Preparing Test Data for Final Evaluation ---")
    try:
        metadata = config.MODEL_METADATA[base_model_name]
        preprocess_function = metadata['preprocess']
        image_size = metadata.get('required_image_size', config.DEFAULT_IMAGE_SIZE)
        
        # Determine canny setting from the run's hyperparameters
        use_canny = hyperparameters.get('use_canny', False)
        print(f"   - Applying Canny Edge based on run config: {use_canny}")

        test_ds, _, _ = data_loader.create_dataset(
            config.TEST_DIR, image_size, config.BATCH_SIZE,
            preprocess_function, is_training=False, use_canny=use_canny
        )
    except Exception as e:
        print(f"❌ Error preparing test data during final evaluation: {e}")
        return None

    # 2. Get predictions
    print("Generating predictions on test set...")
    y_true_list = []
    y_pred_probs_list = []
    
    for images, labels in test_ds:
        preds = model.predict_on_batch(images)
        y_pred_probs_list.append(preds.flatten())
        y_true_list.append(labels.numpy().flatten())
    
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred_probs = np.concatenate(y_pred_probs_list, axis=0)
    
    # Calculate metrics at different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    results = {'test_metrics': {}}
    
    print("\n--- Test Set Metrics at Different Thresholds ---")
    for threshold in thresholds:
        y_pred_classes = (y_pred_probs > threshold).astype(int)
        
        metrics = {
            'threshold': threshold,
            'accuracy': accuracy_score(y_true, y_pred_classes),
            'precision': precision_score(y_true, y_pred_classes, zero_division=0),
            'recall': recall_score(y_true, y_pred_classes, zero_division=0),
            'f1_score': f1_score(y_true, y_pred_classes, zero_division=0)
        }
        
        results['test_metrics'][f'threshold_{threshold}'] = metrics
        print(f"Threshold {threshold}: Acc={metrics['accuracy']:.4f}, "
              f"Prec={metrics['precision']:.4f}, Rec={metrics['recall']:.4f}, "
              f"F1={metrics['f1_score']:.4f}")
    
    # Find optimal threshold
    precisions, recalls, thresholds_pr = precision_recall_curve(y_true, y_pred_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    optimal_idx = np.argmax(f1_scores[:-1])
    optimal_threshold = thresholds_pr[optimal_idx]
    
    y_pred_optimal = (y_pred_probs > optimal_threshold).astype(int)
    
    # Final test metrics
    final_metrics = {
        'test_roc_auc': roc_auc_score(y_true, y_pred_probs),
        'test_pr_auc': average_precision_score(y_true, y_pred_probs),
        'optimal_threshold': optimal_threshold,
        'test_accuracy': accuracy_score(y_true, y_pred_optimal),
        'test_precision': precision_score(y_true, y_pred_optimal, zero_division=0),
        'test_recall': recall_score(y_true, y_pred_optimal, zero_division=0),
        'test_f1_score': f1_score(y_true, y_pred_optimal, zero_division=0)
    }
    
    results['final_test_metrics'] = final_metrics
    
    print(f"\n🎯 FINAL TEST RESULTS:")
    print(f"   ROC AUC: {final_metrics['test_roc_auc']:.4f}")
    print(f"   PR AUC: {final_metrics['test_pr_auc']:.4f}")
    print(f"   Optimal Threshold: {final_metrics['optimal_threshold']:.4f}")
    print(f"   Test Accuracy: {final_metrics['test_accuracy']:.4f}")
    print(f"   Test Precision: {final_metrics['test_precision']:.4f}")
    print(f"   Test Recall: {final_metrics['test_recall']:.4f}")
    print(f"   Test F1 Score: {final_metrics['test_f1_score']:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_optimal)
    print(f"\n📊 Confusion Matrix (Threshold: {optimal_threshold:.3f}):")
    print(f"   TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"   FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    # Classification Report
    print(f"\n📋 Detailed Classification Report:")
    class_names = ['Healed', 'Non-healed']
    print(classification_report(y_true, y_pred_optimal, target_names=class_names, zero_division=0))

    # Additional detailed statistics
    print(f"\n📈 Additional Model Statistics:")
    print(f"   Total Test Samples: {len(y_true)}")
    print(f"   Class Distribution: {np.bincount(y_true.astype(int))} (Healed: {np.sum(y_true == 0)}, Non-healed: {np.sum(y_true == 1)})")
    print(f"   Class Balance: {np.mean(y_true):.1%} non-healed")
    
    # Prediction statistics
    print(f"\n🎯 Prediction Analysis:")
    print(f"   Mean Predicted Probability: {np.mean(y_pred_probs):.4f}")
    print(f"   Std Predicted Probability: {np.std(y_pred_probs):.4f}")
    print(f"   Min Predicted Probability: {np.min(y_pred_probs):.4f}")
    print(f"   Max Predicted Probability: {np.max(y_pred_probs):.4f}")
    
    # Confidence analysis
    confident_predictions = np.sum((y_pred_probs < 0.2) | (y_pred_probs > 0.8))
    print(f"   Confident Predictions (p<0.2 or p>0.8): {confident_predictions}/{len(y_pred_probs)} ({confident_predictions/len(y_pred_probs):.1%})")
    
    # Store data for plotting
    results['plot_data'] = {
        'y_true': y_true,
        'y_pred_probs': y_pred_probs,
        'y_pred_optimal': y_pred_optimal,
        'confusion_matrix': cm
    }
    
    print("="*50)
    print("🎉 Test set evaluation complete!")
    print("="*50)
    return results

def save_final_results(model_name, results, output_dir, filename='final_test_results.png'):
    """
    Saves final results and creates comprehensive plots.
    
    Args:
        model_name (str): Name of the model
        results (dict): Comprehensive evaluation results
        output_dir (str): Directory to save the results and plots.
        filename (str): The filename for the combined plot.
    """
    final_dir = output_dir
    os.makedirs(final_dir, exist_ok=True)  # Ensure directory exists
    
    # Save results
    results_to_save = {k: v for k, v in results.items() if k != 'plot_data'}
    with open(os.path.join(final_dir, 'final_test_results.json'), 'w') as f:
        json.dump(results_to_save, f, indent=4, default=str)
    
    # Create final evaluation plot
    if 'plot_data' in results:
        plot_data = results['plot_data']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Add subplot labels
        subplot_labels = ['(A)', '(B)', '(C)', '(D)']
        for ax, label in zip(axes.flatten(), subplot_labels):
            ax.text(-0.1, 1.05, label, transform=ax.transAxes, fontsize=16, fontweight='400', va='top', ha='right')

        # Confusion Matrix
        sns.heatmap(plot_data['confusion_matrix'], annot=True, fmt='d',
                   cmap='Blues', ax=axes[0,0],
                   xticklabels=['Healed', 'Non-healed'], yticklabels=['Healed', 'Non-healed'])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('Actual')
        
        # ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(plot_data['y_true'], plot_data['y_pred_probs'])
        roc_auc = roc_auc_score(plot_data['y_true'], plot_data['y_pred_probs'])
        axes[0,1].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        axes[0,1].plot([0, 1], [0, 1], 'k--', label='Baseline')
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curve')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(plot_data['y_true'], plot_data['y_pred_probs'])
        pr_auc = average_precision_score(plot_data['y_true'], plot_data['y_pred_probs'])
        axes[1,0].plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.4f})')
        # Add baseline for PR curve
        no_skill = np.sum(plot_data['y_true'] == 1) / len(plot_data['y_true'])
        axes[1,0].axhline(y=no_skill, color='r', linestyle='--', label=f'Baseline ({no_skill:.2f})')
        axes[1,0].set_xlabel('Recall')
        axes[1,0].set_ylabel('Precision')
        axes[1,0].set_title('Precision-Recall Curve')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Prediction Distribution
        axes[1,1].hist(plot_data['y_pred_probs'][plot_data['y_true'] == 0],
                      bins=30, alpha=0.7, label='Healed', color='blue')
        axes[1,1].hist(plot_data['y_pred_probs'][plot_data['y_true'] == 1],
                      bins=30, alpha=0.7, label='Non-healed', color='red')
        axes[1,1].set_xlabel('Predicted Probability')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Prediction Distribution')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(final_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
        # Create individual high-resolution plots
        print("📊 Generating detailed individual plots...")
        
        # Individual Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(plot_data['confusion_matrix'], annot=True, fmt='d',
                   cmap='Blues', cbar_kws={'label': 'Count'},
                   xticklabels=['Healed', 'Non-healed'], yticklabels=['Healed', 'Non-healed'])
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(os.path.join(final_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Individual ROC Curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(plot_data['y_true'], plot_data['y_pred_probs'])
        roc_auc = roc_auc_score(plot_data['y_true'], plot_data['y_pred_probs'])
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Baseline')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(final_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Individual Precision-Recall Curve
        plt.figure(figsize=(8, 6))
        precision, recall, _ = precision_recall_curve(plot_data['y_true'], plot_data['y_pred_probs'])
        pr_auc = average_precision_score(plot_data['y_true'], plot_data['y_pred_probs'])
        plt.plot(recall, precision, linewidth=2, label=f'PR Curve (AUC = {pr_auc:.4f})')
        # Add baseline for PR curve
        no_skill = np.sum(plot_data['y_true'] == 1) / len(plot_data['y_true'])
        plt.axhline(y=no_skill, color='r', linestyle='--', label=f'Baseline ({no_skill:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(final_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Individual Prediction Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(plot_data['y_pred_probs'][plot_data['y_true'] == 0],
                bins=30, alpha=0.7, label='Healed (Class 0)', color='blue', edgecolor='black')
        plt.hist(plot_data['y_pred_probs'][plot_data['y_true'] == 1],
                bins=30, alpha=0.7, label='Non-healed (Class 1)', color='red', edgecolor='black')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title(f'Prediction Distribution - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(final_dir, 'prediction_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Individual plots saved: confusion_matrix.png, roc_curve.png, precision_recall_curve.png, prediction_distribution.png")
    
    print(f"\n📁 All final results and plots saved to: {final_dir}/")
    print(f"📊 Summary: JSON metrics, combined plot, and individual high-res plots generated!")

def test_model(model, test_data, test_dir, model_dir_name, run_id, hyperparameters, output_dir=None):
    """
    Tests the final model on the test dataset and saves evaluation results.
    
    Args:
        model: The fine-tuned model
        test_data: The test dataset
        test_dir: The path to the test directory
        model_dir_name: The name of the model directory (e.g., VGG16 - Dense_ON - Canny_ON)
        run_id: The ID of the specific run (e.g., run_001)
        hyperparameters: The hyperparameters used for this run.
        output_dir (str, optional): An optional output directory to save results.
        
    Returns:
        dict: A dictionary containing test results.
    """
    print(f"\n--- Testing Model: {model_dir_name} (Run: {run_id}) ---")
    
    # 1. Prepare Test Data
    print("\n--- Stage 2: Preparing Test Data ---")
    try:
        metadata = config.MODEL_METADATA[model_dir_name.split(' - ')[0]] # Extract base model name
        preprocess_function = metadata['preprocess']
        image_size = metadata.get('required_image_size', config.DEFAULT_IMAGE_SIZE)
        
        # Determine canny setting from hyperparameters used for the specific run
        use_canny_for_test = model_dir_name.split(' - ')[1].split('_')[1] == 'ON' # Check if Canny is ON
        print(f"   - Applying Canny Edge based on run config: {use_canny_for_test}")
        
        test_ds, _, _ = data_loader.create_dataset(
            test_dir, image_size, config.BATCH_SIZE,
            preprocess_function, is_training=False, use_canny=use_canny_for_test
        )
    except Exception as e:
        print(f"❌ Error preparing test data: {e}")
        return None

    # 2. Final Evaluation
    print("\n--- Stage 3: Final Evaluation on Test Set ---")
    
    # Run final evaluation
    results = evaluate_final_model(model, model_dir_name.split(' - ')[0], hyperparameters)
    
    # Save examples for inspection
    base_dir = output_dir if output_dir else config.RESULTS_BASE_DIR
    final_model_dir_for_examples = os.path.join(base_dir, 'final_model')
    save_preprocessed_examples(test_data, model, final_model_dir_for_examples, model_dir_name)

    # Save final results
    test_results_dir = os.path.join(base_dir, f'test_only_results_{model_dir_name.replace(" ", "_").replace("-", "_")}')
    save_final_results(model_dir_name.split(' - ')[0], results, test_results_dir)
    
    print(f"\n🎉 TESTING COMPLETE for {model_dir_name}")
    print(f"📊 Test AUC: {results['final_test_metrics']['test_roc_auc']:.4f}")
    print(f"📁 Results saved to: {test_results_dir}")

    return {
        'model_dir_name': model_dir_name,
        'run_id': run_id,
        'test_accuracy_optimal': results['final_test_metrics']['test_accuracy'],
        'test_precision_optimal': results['final_test_metrics']['test_precision'],
        'test_recall_optimal': results['final_test_metrics']['test_recall'],
        'test_f1_optimal': results['final_test_metrics']['test_f1_score'],
        'test_auc': results['final_test_metrics']['test_roc_auc'],
        'optimal_threshold': results['final_test_metrics']['optimal_threshold'],
    }


def find_best_overall_run():
    """Finds the best overall model run from the winner summary."""
    summary_path = os.path.join(config.RESULTS_BASE_DIR, 'winner_selection_summary.json')
    if not os.path.exists(summary_path):
        print(f"❌ Cannot proceed: 'winner_selection_summary.json' not found!")
        print("Run the analysis script (analyze_all_models.py) first to generate the summary.")
        return None
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    winner = summary.get('winner')
    if not winner:
        print("❌ Summary file is missing the 'winner' key.")
        return None
        
    print(f"🏆 Found Winner from summary: {winner['model_dir_name']} (Run: {winner['run_id']})")
    return winner



def run_test_for_model(model_info, skip_retraining, args=None):
    """
    Runs the full testing pipeline for a single, identified best model run.
    This includes optional retraining and final evaluation.
    """
    if not model_info:
        print("❌ No model information provided to test.")
        return None

    model_dir_name = model_info['model_dir_name']
    run_id = model_info['run_id']
    hyperparameters = model_info['hyperparameters']
    model_path = model_info['model_path']
    base_model_name = model_info['model_name']

    print("\n" + "="*80)
    print(f"🚀 Preparing to Test Best Model:")
    print(f"   - Model: {model_dir_name}")
    print(f"   - Run ID: {run_id}")
    print(f"   - Original Val AUC: {model_info['val_auc']:.4f}")
    print("="*80 + "\n")

    output_dir = getattr(args, 'output_dir', None) if args else None

    if not skip_retraining:
        print("--- Stage 1: Retraining on combined train+validation data ---")
        
        # Create combined dataset from train + validation
        combined_ds, total_c0, total_c1 = create_combined_dataset(base_model_name, hyperparameters)
        
        # Get custom epochs from args if available
        custom_epochs = getattr(args, 'epochs', None) if args else None
        
        # Fine-tune the winning model on combined data (using total epochs run, not just best epoch)
        model, history = fine_tune_winning_model(
            base_model_name, hyperparameters, combined_ds, 
            total_c0, total_c1, model_info['total_epochs'], model_path, custom_epochs,
            output_dir=output_dir
        )
        
        print("✅ Retraining complete!")

    if skip_retraining:
        print("--- Stage 1: SKIPPED - Loading model weights directly ---")
        try:
            metadata = config.MODEL_METADATA[base_model_name]
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

            # --- FIX ---
            # The model's weights were saved *after* layers were unfrozen for fine-tuning.
            # We must replicate that state before loading the weights.
            print("Preparing model for weight loading by unfreezing layers and compiling...")
            model_builder.unfreeze_layers(model)

            # Compile the model to ensure it's fully built. The optimizer details
            # don't matter here since we are only doing evaluation.
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.AUC(name='auc')]
            )
            # --- END FIX ---

            model.load_weights(model_path)
            print("✅ Model built and weights loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading model with weights: {e}")
            traceback.print_exc()
            return None
    
    # The rest of the function is now handled inside test_model,
    # making this function cleaner and just for orchestration.
    results = test_model(model, None, config.TEST_DIR, model_dir_name, run_id, hyperparameters, output_dir=output_dir)
    
    # Final summary of all outputs
    if results:
        print("\n" + "="*80)
        print("📁 COMPREHENSIVE OUTPUT SUMMARY")
        print("="*80)
        print("Generated files and visualizations:")
        print(f"📊 Retraining History: results/final_model/final_training_history.png")
        print(f"📊 Combined Test Results: results/test_only_results_*/final_test_results.png")
        print(f"📊 Individual Plots:")
        print(f"   • Confusion Matrix: results/test_only_results_*/confusion_matrix.png")
        print(f"   • ROC Curve: results/test_only_results_*/roc_curve.png")
        print(f"   • Precision-Recall Curve: results/test_only_results_*/precision_recall_curve.png") 
        print(f"   • Prediction Distribution: results/test_only_results_*/prediction_distribution.png")
        print(f"📄 JSON Metrics: results/test_only_results_*/final_test_results.json")
        print(f"📄 Training History: results/final_model/final_training_history.json")
        print(f"💾 Final Model Weights: results/final_model/final_fine_tuned_model_weights.h5")
        print("="*80)
        print("✅ FINAL TESTING COMPLETE - All outputs generated!")
        print("="*80)

    return results

def test_final_model_from_folder(folder_path, output_dir=None):
    """
    Loads a final model from a specified folder and evaluates it on the test set.
    Args:
        folder_path (str): Path to the folder containing final_model_details.json and final_fine_tuned_model_weights.h5
        output_dir (str, optional): An optional directory to save the results. If None, results are saved in `folder_path`.
    """
    print(f"\n=== Testing final model from folder: {folder_path} ===")
    details_path = os.path.join(folder_path, 'final_model_details.json')
    weights_path = os.path.join(folder_path, 'final_fine_tuned_model_weights.h5')
    if not os.path.exists(details_path):
        print(f"❌ Could not find final_model_details.json in {folder_path}")
        return
    if not os.path.exists(weights_path):
        print(f"❌ Could not find final_fine_tuned_model_weights.h5 in {folder_path}")
        return
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
    model_builder.unfreeze_layers(model)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )
    model.load_weights(weights_path)
    print("✅ Model built and weights loaded successfully.")
    # Evaluate on test set
    results = evaluate_final_model(model, model_name, hyperparameters)
    
    # Determine output directory
    save_dir = output_dir if output_dir is not None else folder_path
    os.makedirs(save_dir, exist_ok=True)
    
    # Save outputs in the determined folder
    save_final_results(model_name, results, save_dir, filename='final_test_results_with_baseline.png')
    
    print(f"\n🎉 TESTING COMPLETE for final model in {folder_path}")
    print(f"📁 Results saved to: {save_dir}")

def main():
    parser = argparse.ArgumentParser(description="Script to test the best trained models on the final test set.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--winner', action='store_true', help='Find the best overall model from the summary file and test it.')
    group.add_argument('--model', type=str, help='Specify a model name to find and test (e.g., VGG16). Use with --dense/--canny flags.')
    group.add_argument('--all-models', action='store_true', help='Test the best configuration for every model defined in config.py.')
    group.add_argument('--final-model-folder', type=str, help='Directly test a final model in the specified folder (expects final_model_details.json and final_fine_tuned_model_weights.h5).')

    parser.add_argument('--dense', type=str, choices=['ON', 'OFF'], help='Specify the dense layer configuration (ON/OFF). Used with --model.')
    parser.add_argument('--canny', type=str, choices=['ON', 'OFF'], help='Specify the Canny edge configuration (ON/OFF). Used with --model.')
    parser.add_argument('--skip-training', action='store_true', help='Skip the final retraining step and test the saved model directly.')
    parser.add_argument('--output-dir', type=str, help='Optional. Specify a directory to save all output files and folders.')
    
    args = parser.parse_args()

    if args.final_model_folder:
        test_final_model_from_folder(args.final_model_folder, args.output_dir)
        return

    if args.winner:
        print("🏆 Finding the single best model across all runs from summary...")
        model_info = find_best_overall_run()
        if model_info:
            run_test_for_model(model_info, args.skip_training, args)

    elif args.model:
        if not (args.dense and args.canny):
            print("--- Note: No configuration specified. Finding the best overall run for this model. ---")
            print(f"--- Use --dense and --canny to select a specific configuration. ---")
            
        print(f"🔍 Finding best run for specified model: {args.model} (Dense: {args.dense or 'Any'}, Canny: {args.canny or 'Any'})")
        model_info = find_specified_model_run(args.model, args.dense, args.canny)
        if model_info:
            run_test_for_model(model_info, args.skip_training, args)
        else:
            print("❌ No matching model run found.")

    elif args.all_models:
        print("🚀 Testing the best version of ALL models defined in config.py")
        all_results = []
        all_model_names = list(config.MODEL_METADATA.keys())

        for model_name in all_model_names:
            print("\n" + "#"*80)
            print(f"# Testing best configuration for: {model_name}")
            print("#"*80)
            # Find the best configuration for the current model by searching all variants
            model_info = find_specified_model_run(model_name, dense_state=None, canny_state=None)
            
            if model_info:
                test_results = run_test_for_model(model_info, args.skip_training, args)
                if test_results:
                    all_results.append(test_results)
            else:
                print(f"⚠️  No trained runs found for {model_name}. Skipping.")
        
        if all_results:
            print("\n" + "="*80)
            print("🏁 FINAL SUMMARY OF ALL MODELS 🏁")
            print("="*80)
            summary_df = pd.DataFrame(all_results)
            summary_df = summary_df.sort_values(by='test_auc', ascending=False)
            
            # Define columns to display
            display_columns = [
                'model_dir_name', 'run_id', 'test_auc', 'test_f1_optimal', 
                'optimal_threshold', 'test_accuracy_optimal', 
                'test_precision_optimal', 'test_recall_optimal'
            ]
            
            # Ensure all display columns exist in the DataFrame, add them with None if not
            for col in display_columns:
                if col not in summary_df.columns:
                    summary_df[col] = None
            
            print(summary_df.to_string(index=False, columns=display_columns))

            # Save summary to CSV
            summary_csv_path = os.path.join(config.RESULTS_BASE_DIR, 'all_models_test_summary.csv')
            summary_df.to_csv(summary_csv_path, index=False)
            print(f"\n📊 Summary saved to {summary_csv_path}")

if __name__ == '__main__':
    main() 