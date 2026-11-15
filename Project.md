# MasterChef: A Scientific Deep Learning Pipeline for Medical Image Classification

**Objective:** To create a modular, universal, and scientifically rigorous training pipeline in Python using TensorFlow/Keras for the systematic evaluation of deep learning models on medical image classification. This pipeline supports a fair, "apples-to-apples" comparison of multiple pre-trained models, featuring hyperparameter optimization, robust evaluation, and a meticulous final testing protocol. ✅

## Core Methodology


The project follows a multi-phase approach designed for reproducibility and scientific rigor.

1.  **Systematic Training & Hyperparameter Search:** ✅
    *   **10 Base Models, 20 Configurations:** Ten different pre-trained models from Keras Applications were selected. Each model was trained in two configurations: one with raw images and one with Canny edge detection applied as a preprocessing step, resulting in 20 total experimental tracks.
    *   **Random Search:** For each of the 20 configurations, a 10-iteration random search was performed to find the optimal hyperparameters.
    *   **Two-Stage Training:**
        1.  **Feature Extraction (50 Epochs):** The base model's weights were frozen, and only the classification head was trained with a learning rate of `1e-3`.
        2.  **Fine-Tuning (50 Epochs):** The top 20% of the base model's layers were unfrozen and trained with a smaller, optimized learning rate.
    *   **Validation:** The best hyperparameters for each model configuration were determined by the highest achieved **Validation AUC** during the fine-tuning stage.

2.  **Winner Selection:** ✅
    *   The `final_testing.py` script automatically scans the results of all 20 model configurations and their corresponding hyperparameter search runs.
    *   The single "winner" is the model configuration and hyperparameter set that achieved the absolute highest **Validation AUC** across all experiments.

3.  **Final Evaluation Protocol:** ✅
    *   **Combined Dataset:** The original training and validation datasets were merged to create a larger dataset for final model training.
    *   **Retraining:** The winning model architecture was instantiated with its best-performing hyperparameters and retrained on this combined dataset. A learning rate 10x smaller than the best fine-tuning learning rate was used for stability.
    *   **Final Testing:** The retrained model was then evaluated on a completely held-out **test dataset** to provide an unbiased assessment of its performance on unseen data.

---

## Technical Implementation Details

This section breaks down the specific configurations and logic implemented in the `src` scripts.

### 1. `config.py` - Central Configuration

This file centralizes all constants and hyperparameters to ensure consistency.

*   **Models Evaluated:** ✅
    *   VGG16, VGG19, ResNet152, DenseNet201, EfficientNetB0, EfficientNetB3, MobileNetV3Large, InceptionV3, Xception, InceptionResNetV2.
*   **Image Preprocessing:** ✅
    *   `DEFAULT_IMAGE_SIZE`: `(256, 256)`.
    *   **Model-Specific Sizes:** VGG16 and VGG19 require `(224, 224)`; InceptionV3, Xception, and InceptionResNetV2 require `(299, 299)`.
    *   `USE_LETTERBOX_PADDING`: `True` (maintains aspect ratio).
    *   `USE_CANNY_EDGE`: Toggled between `True` and `False` to create two versions of each model.
*   **Training Parameters:** ✅
    *   `BATCH_SIZE`: `16`
    *   `OPTIMIZER_TYPE`: `Adam`
    *   `LEARNING_RATE_FEATURE_EXTRACTION`: `1e-3`
    *   `LABEL_SMOOTHING_FACTOR`: `0.1`
    *   `FINE_TUNE_LAYER_PERCENTAGE`: `0.2` (top 20% of layers unfrozen).
    *   `EPOCHS_FEATURE_EXTRACTION`: `50`
    *   `EPOCHS_FINE_TUNE`: `50`
    *   `EARLY_STOPPING_PATIENCE`: `10` (monitors `val_auc`).
*   **Hyperparameter Search Space:** ✅
    *   `learning_rate_fine_tune`: `[1e-4, 5e-5, 1e-5, 5e-6, 1e-6]`
    *   `dropout_rate`: `(0.3, 0.7)`
    *   `kernel_regularizer_l2`: `(1e-5, 1e-3)` (log-uniform)
    *   `lr_scheduler_factor`: `[0.1, 0.2, 0.5]`
    *   `lr_scheduler_patience`: `[3, 5, 7]`

### 2. `data_loader.py` - Medical Image Data Pipeline

*   **Conservative Augmentation (for training images):** ✅
    *   `RandomFlip`: `"horizontal"`
    *   `RandomRotation`: `0.06` (±3.4°)
    *   `RandomZoom`: `0.06` (±6%)
    *   `RandomContrast`: `0.04`
    *   `RandomBrightness`: `0.04`
*   **Class Imbalance Handling:** ✅
    *   **Random Oversampling:** During training, `tf.data.experimental.sample_from_datasets` was used to create batches with a target minority class ratio of `0.3`, ensuring the model sees a balanced mix of classes without generating synthetic samples. The perccentage of healed and not healed were about ~30%/~70%, so this is why we used minority class ratio of 0.3 to have stabile class balance.
    *   **Natural Distribution:** For validation, testing, and the final retraining phase, the natural class distribution was used to get a realistic measure of performance.
*   **Model-Specific Preprocessing:** Crucially, after augmentation, each image is passed through the specific `preprocess_input` function associated with its model architecture (e.g., `resnet.preprocess_input`, `efficientnet.preprocess_input`). This ensures that the input data is scaled and normalized exactly as the pre-trained model expects.
*   **Preprocessing Order:** The pipeline strictly follows this sequence for every image: ✅
    1.  **Load Image:** The raw image is loaded from disk.
    2.  **Resize & Pad:** The image is resized to the target dimensions using letterboxing to preserve the aspect ratio.
    3.  **Augmentation (Training Only):** The conservative data augmentation is applied.
    4.  **Canny Edge Detection (If Enabled):** Canny edge detection is performed on the raw (and potentially augmented) image.
    5.  **Model-Specific Normalization:** The appropriate `preprocess_input` function is called last to scale pixel values to the required range (e.g., `[-1, 1]` or `[0, 1]`).

### 3. `model.py` - Universal Model Builder

*   **Adaptive Architecture:** The script creates a standard classification head that attaches to any Keras base model.
*   **Consistent Head:**
    *   `GlobalAveragePooling2D`
    *   (Optional -> Didnt use this (so UseDense256=False)) `Dense` layer with `256` units and L2 regularization (`USE_DENSE_LAYER` flag).
    *   `Dropout`
    *   `Dense` output layer with `1` unit and `sigmoid` activation for binary classification.

### 4. `train.py` - Training Orchestrator

*   Manages the two-stage training protocol and the hyperparameter random search loop.
*   Saves all results, logs, and model weights for each run into a structured directory under `results/`.
*   Monitors `val_auc` to save the best model weights during the fine-tuning phase.

### 5. `final_testing.py` - Final Evaluation Protocol

*   **Winner Identification:** Scans all `history.json` files from all training runs to find the model configuration with the highest `val_auc`.
*   **Retraining Logic:**
    *   Combines the `train` and `validation` directories.
    *   Initializes the winning model with its best hyperparameters.
    *   Loads the weights from the winning validation run.
    *   Continues training on the combined dataset with a learning rate of `best_lr * 0.1`.
*   **Test Evaluation:**
    *   Loads the final, retrained model.
    *   Generates predictions on the unseen test set.
    *   Calculates a full suite of metrics (AUC, F1, Precision, Recall, etc.).
    *   Identifies the optimal classification threshold based on the best F1-score.
    *   Saves comprehensive JSON reports and visualizations (Confusion Matrix, ROC Curve, etc.).

---

## How to Use This Pipeline

### Phase 1: Train Models & Find Best Hyperparameters
Run the training script for each model. This will create a directory for each model configuration and store the results of the 10 hyperparameter search runs inside.
```bash
# Example for one model (repeat for all)
python -m src.train --model EfficientNetB3
```
*You would need to run this twice for each model: once with `USE_CANNY_EDGE=True` in `config.py`, and once with it set to `False`.*

### Phase 2: Select Winner, Retrain, and Test
After all models have been trained, run the final testing script. It will automatically find the best-performing model across all runs, retrain it on the combined data, and evaluate it on the test set.
```bash
# Automatically find winner, retrain, and test
python -m src.final_testing --winner
```

### Phase 3: Analyze Results
The final results, including metrics and plots for the winning model, will be saved in the `results/final_model/` and `results/test_only_results.../` directories. These provide an unbiased performance evaluation suitable for academic reporting.
