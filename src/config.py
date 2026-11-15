# src/config.py

# This file acts as a single source of truth for all constants and hyperparameters
# used in the training pipeline. This ensures consistency and reproducibility
# across different experiments and models.



# --- Required imports for model metadata ---
from tensorflow.keras.applications import (
    VGG16, VGG19, ResNet152, DenseNet201, EfficientNetB0, EfficientNetB3,
    MobileNetV3Large, InceptionV3, Xception, InceptionResNetV2,
    vgg16, vgg19, resnet, densenet, efficientnet, mobilenet_v3,
    inception_v3, xception, inception_resnet_v2
)

# --- Model Configuration ---

# Model Metadata Dictionary
# This dictionary is the key to handling multiple models with different
# requirements in a universal pipeline. It stores:
#   - 'loader': The function to load the base model.
#   - 'preprocess': The specific preprocessing function required by the model.
#   - 'required_image_size' (optional): A specific image size the model must use.
MODEL_METADATA = {
    # --- Standard Keras Applications ---
    "VGG16": {"loader": VGG16, "preprocess": vgg16.preprocess_input, "required_image_size": (224, 224)},
    "VGG19": {"loader": VGG19, "preprocess": vgg19.preprocess_input, "required_image_size": (224, 224)},
    "ResNet152": {"loader": ResNet152, "preprocess": resnet.preprocess_input},
    "DenseNet201": {"loader": DenseNet201, "preprocess": densenet.preprocess_input},
    "EfficientNetB0": {"loader": EfficientNetB0, "preprocess": efficientnet.preprocess_input},
    "EfficientNetB3": {"loader": EfficientNetB3, "preprocess": efficientnet.preprocess_input},
    "MobileNetV3Large": {"loader": MobileNetV3Large, "preprocess": mobilenet_v3.preprocess_input},
    
    # --- Models with Required Input Sizes ---
    "InceptionV3": {"loader": InceptionV3, "preprocess": inception_v3.preprocess_input, "required_image_size": (299, 299)},
    "Xception": {"loader": Xception, "preprocess": xception.preprocess_input, "required_image_size": (299, 299)},
    "InceptionResNetV2": {"loader": InceptionResNetV2, "preprocess": inception_resnet_v2.preprocess_input, "required_image_size": (299, 299)},

    # --- Models from other libraries (example structure) ---
    # "CoAtNet": {"loader": keras_cv.models.CoAtNet, "preprocess": lambda x: x}, # KerasCV preprocess often is just scaling
    # "DeiT": {
    #     "loader": lambda **kwargs: TFAutoModel.from_pretrained("facebook/deit-base-distilled-patch16-224", from_pt=True, **kwargs),
    #     "preprocess": lambda x: AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")(images=x, return_tensors="tf")['pixel_values'],
    #     "required_image_size": (224, 224)
    # },
}




# --- Data & Preprocessing Constants ---

# Default image size for models that do not have a specific requirement.
# Models will be trained on images resized to this dimension.
DEFAULT_IMAGE_SIZE = (256, 256)

# Interpolation method to use when resizing images.
# 'bicubic' is often a good choice for downsampling.
RESIZE_METHOD = 'bicubic'

# If True, images will be resized while maintaining aspect ratio,
# with padding added to fill the target dimensions ("letterboxing").
# If False, images will be stretched to the target dimensions.
USE_LETTERBOX_PADDING = True


# --- Data Paths ---
# Using absolute paths to the dataset location to avoid copying data.
# This points directly to your dataset folder.
TRAIN_DIR = '../../Dataset/EP_Versio3_Removed_Turhat_Lapiammuttu_BadViistot_Proteesi_Levy/train'
VALIDATION_DIR = '../../Dataset/EP_Versio3_Removed_Turhat_Lapiammuttu_BadViistot_Proteesi_Levy/validation'
TEST_DIR = '../../Dataset/EP_Versio3_Removed_Turhat_Lapiammuttu_BadViistot_Proteesi_Levy/test'


# --- Training Constants ---

# Number of samples processed before the model is updated.
BATCH_SIZE = 16

# Number of output classes. For binary classification (healed vs. non-healed), this is 1.
NUM_CLASSES = 1

# Type of optimizer to use during training.
OPTIMIZER_TYPE = 'Adam'

# Initial learning rate for the feature extraction phase (training the head).
LEARNING_RATE_FEATURE_EXTRACTION = 1e-3

# Factor for label smoothing, a regularization technique to prevent
# the model from becoming overconfident. A value of 0.1 is a common choice.
LABEL_SMOOTHING_FACTOR = 0.1

# --- NEW: Oversampling Strategy ---
# This ratio controls the level of balancing for the minority class.
# Its behavior depends on the USE_SMOTE flag:
# - If USE_SMOTE = True: This is the desired ratio of minority to majority samples
#   after SMOTE is applied. E.g., 0.75 means the number of minority samples
#   will be 75% of the number of majority samples. 1.0 means a 50/50 balance.
# - If USE_SMOTE = False: This is the desired proportion of the minority class
#   in each training batch. E.g., 0.4 means batches will contain ~40% minority
#   samples.
# A value less than 1.0 (for SMOTE) or 0.5 (for weighted sampling) helps
# prevent the model from over-correcting its bias.
OVERSAMPLING_MINORITY_CLASS_RATIO = 0.3

# --- NEW: SMOTE Toggle ---
# If True, the data loader will use SMOTE to balance the training set.
# This requires loading the entire dataset into memory and can be slow.
# If False, it will use the `OVERSAMPLING_MINORITY_CLASS_RATIO` strategy.
USE_SMOTE = False


# --- Fixed Head Architecture ---

# Number of units in the dense layer of the classification head.
# This is fixed for all models to ensure a fair comparison.
FIXED_DENSE_UNITS = 256

# --- NEW: Toggle for the dense layer ---
# If False, the architecture will be CNN > GAP > Dropout > Output, inspired by GoogleNet/NiN.
# If True, the architecture is CNN > GAP > Dense(256) > Dropout > Output.
USE_DENSE_LAYER = False

# --- NEW: Toggle for Canny Edge Detection ---
# If True, applies Canny edge detection to the images before any other preprocessing.
# This tests whether models perform better on edge-maps instead of raw images.
USE_CANNY_EDGE = True

# --- Fine-Tuning Strategy ---

# Percentage of layers to unfreeze from the top of the base model for fine-tuning.
# For example, 0.2 means the top 20% of the layers will be made trainable.
FINE_TUNE_LAYER_PERCENTAGE = 0.2

# --- Epoch & Callback Constants ---

# Number of epochs for the initial feature extraction (head training) phase.
EPOCHS_FEATURE_EXTRACTION = 50

# Number of epochs for the fine-tuning phase.
EPOCHS_FINE_TUNE = 50

# Number of epochs with no improvement after which training will be stopped
# to prevent overfitting and save time.
EARLY_STOPPING_PATIENCE = 10

# --- Experiment Constants ---

# Number of iterations for the random hyperparameter search.
# For each model, this many different sets of hyperparameters will be tried.
RANDOM_SEARCH_ITERATIONS = 10

# Seed for random number generators to ensure reproducibility of results.
SEED = 42069

# --- Results & File Organization ---

# Base directory for storing experiment results
RESULTS_BASE_DIR = 'results'

# Standard subdirectory names for logs
LOG_SUBDIRS = {
    'feature_extraction': 'logs_fe',
    'fine_tuning': 'logs_ft'
}

# Standard file names used throughout the pipeline
MODEL_CHECKPOINT_FILENAME = 'best_model_weights.h5'
HYPERPARAMETERS_FILENAME = 'hyperparameters.json'
HISTORY_FILENAME = 'history.json'

# --- Callback Monitor Settings ---

# Metric to monitor for model checkpointing (save best model)
CHECKPOINT_MONITOR = 'val_auc'
CHECKPOINT_MODE = 'max'

# Metric to monitor for early stopping
EARLY_STOPPING_MONITOR = 'val_auc'
EARLY_STOPPING_MODE = 'max'

# Metric to monitor for learning rate reduction
LR_SCHEDULER_MONITOR = 'val_auc'
LR_SCHEDULER_MODE = 'max'
LR_SCHEDULER_MIN_LR = 1e-7




# --- Hyperparameter Search Space (Variable Parameters) ---

# This dictionary defines the search space for hyperparameters that will be
# optimized during the random search.
HYPERPARAMETER_SPACE = {
    # Range of learning rates to try for the fine-tuning phase.
    'learning_rate_fine_tune': [1e-4, 5e-5, 1e-5, 5e-6, 1e-6],
    # Range for the dropout rate in the classification head.
    'dropout_rate': (0.3, 0.7),
    # Range for L2 regularization strength on the dense layer kernels.
    'kernel_regularizer_l2': (1e-5, 1e-3),
    # Factor by which the learning rate will be reduced by ReduceLROnPlateau.
    'lr_scheduler_factor': [0.1, 0.2, 0.5],
    # Number of epochs with no improvement after which learning rate will be reduced.
    'lr_scheduler_patience': [3, 5, 7]
} 