import argparse
import os
import random
import json
import sys
import traceback
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, losses
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau

# Note: KerasCV and Hugging Face Transformers need to be installed.
# pip install keras-cv
# pip install transformers
# import keras_cv
# from transformers import TFAutoModel, AutoImageProcessor

from . import config
from . import data_loader
from . import model as model_builder

# Set the global seed for reproducibility
tf.random.set_seed(config.SEED)
np.random.seed(config.SEED)
random.seed(config.SEED)

def sample_hyperparameters():
    """Samples a set of hyperparameters from the space defined in config.py."""
    try:
        space = config.HYPERPARAMETER_SPACE
        
        # Start with a base set of hyperparameters
        hp = {
            'learning_rate_fine_tune': random.choice(space['learning_rate_fine_tune']),
            'dropout_rate': random.uniform(*space['dropout_rate']),
            'lr_scheduler_factor': random.choice(space['lr_scheduler_factor']),
            'lr_scheduler_patience': random.choice(space['lr_scheduler_patience']),
        }
        
        # Only sample kernel regularizer if the dense layer is actually being used
        if config.USE_DENSE_LAYER:
            hp['kernel_regularizer_l2'] = float(10**random.uniform(np.log10(space['kernel_regularizer_l2'][0]), np.log10(space['kernel_regularizer_l2'][1])))
        else:
            # If no dense layer, L2 regularization is not applicable. Set to 0.
            hp['kernel_regularizer_l2'] = 0.0
            
        return hp

    except KeyError as e:
        raise ValueError(f"Missing hyperparameter configuration in config.py: {e}")
    except (IndexError, ValueError) as e:
        raise ValueError(f"Invalid hyperparameter space configuration: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error sampling hyperparameters: {e}")

def main(args):
    """Main training script."""
    
    if args.every_model:
        all_model_names = list(config.MODEL_METADATA.keys())
        models_to_skip = {"VGG16", "VGG19", "InceptionResNetV2"}  # Use a set for efficient lookup
        model_names = [name for name in all_model_names if name not in models_to_skip]
        
        print(f"--- Starting training for {len(model_names)} models sequentially (skipping {len(models_to_skip)}) ---")
        print(f"Models to train: {model_names}")
        print(f"Skipped models: {sorted(list(models_to_skip))}")
    else:
        model_names = [args.model]

    for model_name in model_names:
        print("\n" + "="*80)
        print(f"🚀 Starting training process for model: {model_name}")
        print("="*80 + "\n")

        try:
            if model_name not in config.MODEL_METADATA:
                available_models = list(config.MODEL_METADATA.keys())
                raise ValueError(f"Model '{model_name}' not defined in MODEL_METADATA. "
                                 f"Available models: {available_models}")
            
            # Determine the model directory name based on the dense layer and canny config
            use_dense_layer = config.USE_DENSE_LAYER
            use_canny = config.USE_CANNY_EDGE
            dense_str = f"Dense_{'ON' if use_dense_layer else 'OFF'}"
            canny_str = f"Canny_{'ON' if use_canny else 'OFF'}"
            model_dir_name = f"{model_name} - {dense_str} - {canny_str}"
            
            print(f"--- Starting training for model configuration: {model_dir_name} ---")

            # 1. Look up model metadata and determine image size
            metadata = config.MODEL_METADATA[model_name]
            image_size = metadata.get('required_image_size', config.DEFAULT_IMAGE_SIZE)
            input_shape = (*image_size, 3)
            print(f"Using image size: {image_size}")

            # 2. Create data loaders
            print("Loading datasets...")
            try:
                preprocess_function = metadata['preprocess']
                train_ds, num_c0, num_c1 = data_loader.create_dataset(
                    config.TRAIN_DIR, image_size, config.BATCH_SIZE, 
                    preprocess_function, is_training=True, use_canny=use_canny
                )
                val_ds, _, _ = data_loader.create_dataset(
                    config.VALIDATION_DIR, image_size, config.BATCH_SIZE, 
                    preprocess_function, is_training=False, use_canny=use_canny
                )
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Dataset directory not found: {e}")
            except Exception as e:
                raise RuntimeError(f"Error loading datasets: {e}")
            
            # With a balanced dataset from the data_loader, we define an epoch as
            # seeing a number of samples equivalent to twice the size of the larger class.
            # This ensures the model sees a rich mix of both classes.
            if num_c0 == 0 or num_c1 == 0:
                raise ValueError("Training requires samples from both classes, but one class is empty.")
            
            # The data loader now handles balancing, so class weights are not needed.
            steps_per_epoch = (max(num_c0, num_c1) * 2) // config.BATCH_SIZE
            
            print(f"Training samples: Class 0: {num_c0}, Class 1: {num_c1}")
            print(f"Using balanced data generator. Steps per epoch: {steps_per_epoch}")

            # 3. Random Search Loop
            for i in range(config.RANDOM_SEARCH_ITERATIONS):
                run_num = i + 1
                print(f"\n--- Starting Random Search Run {run_num}/{config.RANDOM_SEARCH_ITERATIONS} for {model_dir_name} ---")

                try:
                    # Create unique results directory for this run
                    run_dir = os.path.join(config.RESULTS_BASE_DIR, model_dir_name, f'run_{run_num}')
                    try:
                        os.makedirs(run_dir, exist_ok=True)
                    except OSError as e:
                        raise OSError(f"Failed to create results directory '{run_dir}': {e}")

                    # Sample hyperparameters
                    try:
                        hp = sample_hyperparameters()
                        hp['use_dense_layer'] = use_dense_layer # Explicitly add architecture flag
                        hp['use_canny'] = use_canny # Explicitly add canny flag
                        
                        print("Sampled Hyperparameters:")
                        for key, val in hp.items():
                            if key in ['use_dense_layer', 'use_canny']:
                                print(f"  - {key}: {val}")
                            elif isinstance(val, float):
                                print(f"  - {key}: {val:.7f}")
                            else:
                                print(f"  - {key}: {val}")
                    except Exception as e:
                        print(f"❌ Error sampling hyperparameters for run {run_num}: {e}")
                        continue

                    # Save hyperparameters
                    try:
                        # Convert numpy types to Python types for JSON serialization
                        hp_serializable = {}
                        for key, val in hp.items():
                            if isinstance(val, (np.floating, np.integer)):
                                hp_serializable[key] = float(val)
                            else:
                                hp_serializable[key] = val
                        
                        with open(os.path.join(run_dir, config.HYPERPARAMETERS_FILENAME), 'w') as f:
                            json.dump(hp_serializable, f, indent=4)
                    except Exception as e:
                        print(f"⚠️  Warning: Failed to save hyperparameters for run {run_num}: {e}")
                    
                    # Build the model
                    try:
                        print("Building model...")
                        model = model_builder.build_model(
                            base_model_loader=metadata['loader'],
                            input_shape=input_shape,
                            dense_units=config.FIXED_DENSE_UNITS,
                            dropout_rate=hp['dropout_rate'],
                            kernel_regularizer_l2=hp['kernel_regularizer_l2'],
                            use_dense_layer=hp['use_dense_layer']
                        )
                    except Exception as e:
                        print(f"❌ Error building model for run {run_num}: {e}")
                        traceback.print_exc()
                        continue

                    # --- Two-Stage Training ---

                    # === Stage 1: Head Training (Feature Extraction) ===
                    try:
                        print("\n--- Stage 1: Feature Extraction (Training Head) ---")
                        model.compile(
                            optimizer=optimizers.Adam(learning_rate=config.LEARNING_RATE_FEATURE_EXTRACTION),
                            loss=losses.BinaryCrossentropy(label_smoothing=config.LABEL_SMOOTHING_FACTOR),
                            metrics=[tf.keras.metrics.AUC(name='auc')]
                        )
                        
                        callbacks_fe = [
                            TensorBoard(log_dir=os.path.join(run_dir, config.LOG_SUBDIRS['feature_extraction'])),
                            EarlyStopping(monitor='val_loss', patience=config.EARLY_STOPPING_PATIENCE, verbose=1, restore_best_weights=True)
                        ]
                        
                        model.fit(
                            train_ds,
                            epochs=config.EPOCHS_FEATURE_EXTRACTION,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=val_ds,
                            callbacks=callbacks_fe
                        )
                        print("✅ Feature extraction phase completed successfully")
                    except Exception as e:
                        print(f"❌ Error in feature extraction phase for run {run_num}: {e}")
                        traceback.print_exc()
                        continue

                    # === Stage 2: Fine-Tuning ===
                    try:
                        print("\n--- Stage 2: Fine-Tuning ---")
                        model_builder.unfreeze_layers(model)
                        
                        model.compile(
                            optimizer=optimizers.Adam(learning_rate=hp['learning_rate_fine_tune']),
                            loss=losses.BinaryCrossentropy(label_smoothing=config.LABEL_SMOOTHING_FACTOR),
                            metrics=[tf.keras.metrics.AUC(name='auc')]
                        )
                        
                        # Create callbacks, keeping a reference to EarlyStopping
                        early_stopping_callback_ft = EarlyStopping(monitor=config.EARLY_STOPPING_MONITOR, mode=config.EARLY_STOPPING_MODE, patience=config.EARLY_STOPPING_PATIENCE, verbose=1)

                        callbacks_ft = [
                            ModelCheckpoint(
                                filepath=os.path.join(run_dir, config.MODEL_CHECKPOINT_FILENAME),
                                monitor=config.CHECKPOINT_MONITOR,
                                mode=config.CHECKPOINT_MODE,
                                save_best_only=True,
                                save_weights_only=True,
                                verbose=1
                            ),
                            TensorBoard(log_dir=os.path.join(run_dir, config.LOG_SUBDIRS['fine_tuning'])),
                            early_stopping_callback_ft,
                            ReduceLROnPlateau(
                                monitor=config.LR_SCHEDULER_MONITOR,
                                mode=config.LR_SCHEDULER_MODE,
                                factor=hp['lr_scheduler_factor'],
                                patience=hp['lr_scheduler_patience'],
                                verbose=1,
                                min_lr=config.LR_SCHEDULER_MIN_LR
                            )
                        ]

                        history = model.fit(
                            train_ds,
                            epochs=config.EPOCHS_FINE_TUNE,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=val_ds,
                            callbacks=callbacks_ft
                        )
                        print("✅ Fine-tuning phase completed successfully")
                    except Exception as e:
                        print(f"❌ Error in fine-tuning phase for run {run_num}: {e}")
                        traceback.print_exc()
                        continue
                    
                    # Save final history and metadata
                    try:
                        # Convert Keras history to be JSON serializable
                        history_log = {k: [float(v) for v in vals] for k, vals in history.history.items()}

                        # Get best epoch details from the fine-tuning history
                        monitor_metric = config.EARLY_STOPPING_MONITOR
                        monitor_mode = config.EARLY_STOPPING_MODE
                        
                        best_epoch_num = 0
                        best_val_auc = 0
                        best_val_loss = 0

                        # Ensure the monitored metric exists in the history to avoid errors
                        if monitor_metric in history_log and history_log[monitor_metric]:
                            if monitor_mode == 'max':
                                best_epoch_idx = np.argmax(history_log[monitor_metric])
                            else:
                                best_epoch_idx = np.argmin(history_log[monitor_metric])
                            
                            best_epoch_num = int(best_epoch_idx + 1)
                            # Safe retrieval of metrics from the best epoch
                            best_val_auc = history_log.get('val_auc', [0])[best_epoch_idx] if 'val_auc' in history_log else 0
                            best_val_loss = history_log.get('val_loss', [0])[best_epoch_idx] if 'val_loss' in history_log else 0
                        
                        # The `stopped_epoch` attribute is 0 if training was not stopped early.
                        stopped_epoch_val = early_stopping_callback_ft.stopped_epoch
                        
                        # Create a comprehensive summary dictionary for the run
                        run_summary = {
                            "model_config": {
                                "model_name": model_name,
                                "model_dir_name": model_dir_name,
                                "use_dense_layer": hp['use_dense_layer'],
                                "use_canny": hp['use_canny']
                            },
                            "hyperparameters": hp_serializable,
                            "training_summary": {
                                "total_epochs_ran": len(history_log.get('loss', [])),
                                "early_stopping_monitor": monitor_metric,
                                "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
                                "stopped_epoch": stopped_epoch_val,
                                "best_epoch": best_epoch_num,
                                "best_val_auc": best_val_auc,
                                "best_val_loss": best_val_loss
                            },
                            "keras_history": history_log
                        }

                        with open(os.path.join(run_dir, config.HISTORY_FILENAME), 'w') as f:
                            json.dump(run_summary, f, indent=4)
                        print("✅ Detailed training history and summary saved successfully")
                    except Exception as e:
                        print(f"⚠️  Warning: Failed to save detailed training history for run {run_num}: {e}")
                    
                    print(f"✅ Finished Run {run_num}/{config.RANDOM_SEARCH_ITERATIONS} for {model_dir_name}")

                except KeyboardInterrupt:
                    print(f"\n⚠️  Training interrupted by user during run {run_num}")
                    print("Cleaning up and exiting...")
                    raise # Re-raise to be caught by outer handler
                except Exception as e:
                    print(f"❌ Unexpected error in run {run_num}: {e}")
                    traceback.print_exc()
                    print(f"Continuing to next run...")
                    continue

            print(f"\n--- All runs for {model_dir_name} complete. Results are in the '{config.RESULTS_BASE_DIR}/{model_dir_name}' directory. ---")

        except KeyboardInterrupt:
            # This allows the user to stop the entire training process
            print(f"\n⚠️  Training for '{model_name}' interrupted by user.")
            raise

        except Exception as e:
            # This catches errors during setup for a specific model
            print(f"❌ Failed to process model '{model_name}': {e}")
            traceback.print_exc()
            if args.every_model:
                print(f"--- Skipping to the next model. ---")
                continue
            else:
                # If only one model was specified, exit on setup failure
                sys.exit(1)

    if args.every_model:
        print("\n" + "="*80)
        print(f"✅✅✅ All model training processes have been completed. ✅✅✅")
        print("="*80)

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description="Universal model training pipeline.")
        model_group = parser.add_mutually_exclusive_group(required=True)
        model_group.add_argument(
            '--model',
            type=str,
            help="The name of the model to train. Check config.py for available models."
        )
        model_group.add_argument(
            '--every-model',
            action='store_true',
            help="Train every model defined in config.py sequentially."
        )
        
        # Example usage:
        # python -m src.train --model VGG16
        # python -m src.train --every-model
        # Note: Assumes you run this from the 'AA - MasterChef' directory and have created a 'data' dir.
        # You might need to set your PYTHONPATH=. or run as a module.
        
        # To test, you would need to create dummy data directories:
        # E.g., mkdir -p data/train data/validation
        # And add some dummy images like `dummy_0.jpg` and `dummy_1.jpg` to both.
        
        parsed_args = parser.parse_args()
        main(parsed_args)
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1) 