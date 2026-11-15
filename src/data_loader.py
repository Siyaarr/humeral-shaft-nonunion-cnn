import tensorflow as tf
import os
import numpy as np
import random
import cv2
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
from . import config

def resize_and_pad_image(image, target_size, resize_method, use_letterbox):
    """
    Resizes an image to a target size, with an option to maintain aspect ratio
    by padding (letterboxing).

    Args:
        image (tf.Tensor): The input image tensor.
        target_size (tuple): The target (height, width).
        resize_method (str): The interpolation method for resizing.
        use_letterbox (bool): Whether to use letterboxing.

    Returns:
        tf.Tensor: The resized and padded image.
    """
    target_height, target_width = target_size
    if not use_letterbox:
        return tf.image.resize(image, [target_height, target_width], method=resize_method)

    img_shape = tf.shape(image)
    img_h, img_w = img_shape[0], img_shape[1]

    aspect_ratio_image = tf.cast(img_w, tf.float32) / tf.cast(img_h, tf.float32)
    aspect_ratio_target = tf.cast(target_width, tf.float32) / tf.cast(target_height, tf.float32)

    if aspect_ratio_image > aspect_ratio_target:
        new_w = target_width
        new_h = tf.cast(target_width / aspect_ratio_image, tf.int32)
    else:
        new_h = target_height
        new_w = tf.cast(target_height * aspect_ratio_image, tf.int32)

    image = tf.image.resize(image, [new_h, new_w], method=resize_method)

    pad_h = target_height - new_h
    pad_w = target_width - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    paddings = [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
    image = tf.pad(image, paddings, mode='CONSTANT', constant_values=0)
    
    # Final resize to ensure exact target dimensions, in case of rounding errors
    image = tf.image.resize(image, [target_height, target_width], method=resize_method)
    return image

def get_label_from_filename(file_path):
    """
    Extracts the binary label from a filename.
    Handles patterns like '*_0.jpg', '*_1.jpg', or 'name._0.jpg', 'name._1.jpg'.
    """
    filename = tf.strings.split(file_path, os.path.sep)[-1]
    # Remove the file extension first
    name_without_ext = tf.strings.split(filename, '.jpg')[0]
    
    # Look for '_0' or '_1' pattern in the filename
    if tf.strings.regex_full_match(name_without_ext, r'.*_0$'):
        return tf.constant(0, dtype=tf.int32)
    elif tf.strings.regex_full_match(name_without_ext, r'.*_1$'):
        return tf.constant(1, dtype=tf.int32)
    else:
        # Fallback: try to extract the last part after underscore
        parts = tf.strings.split(name_without_ext, '_')
        label_str = parts[-1]
        return tf.strings.to_number(label_str, out_type=tf.int32)

# Conservative data augmentation for medical X-ray images
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.06, fill_mode='constant', fill_value=0.0),  
    tf.keras.layers.RandomZoom(0.06, fill_mode='constant', fill_value=0.0),      
    tf.keras.layers.RandomContrast(0.04),                                       
    tf.keras.layers.RandomBrightness(0.04),                                    
])

# --- Canny Edge Detection Functions ---

def canny_edge_detection(image_tensor):
    """Applies Canny edge detection using OpenCV."""
    # This function is designed to be wrapped by tf.py_function, so it gets a tensor.
    image_np = image_tensor.numpy()
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    # Apply a blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply Canny edge detection with specified thresholds
    edges = cv2.Canny(blurred, 25, 55)
    # Convert the single-channel edge map back to a 3-channel image for model compatibility
    edges_3_channel = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return edges_3_channel.astype(np.float32)

@tf.function
def apply_canny_tf(image, height, width):
    """Wraps the OpenCV Canny function for use in a tf.data pipeline."""
    # tf.py_function runs a Python function on a tensor.
    [edge_image] = tf.py_function(canny_edge_detection, [image], [tf.float32])
    # The output shape must be explicitly set.
    edge_image.set_shape([height, width, 3])
    return edge_image

# --- Main Data Processing Functions ---

def process_path(file_path, label, image_size, preprocess_function, apply_aug, use_canny):
    """
    Loads, preprocesses, resizes, and augments a single image.
    """
    # 1. Load Image
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)

    # 2. Apply "letterboxing" resize/pad first on the raw image.
    image = resize_and_pad_image(image, image_size, config.RESIZE_METHOD, config.USE_LETTERBOX_PADDING)

    # 3. Conditionally apply data augmentation.
    # This should be done on the [0, 255] range image BEFORE Canny or model-specific preprocessing.
    if apply_aug:
        image = data_augmentation(image, training=True)

    # 4. Conditionally apply Canny Edge Detection.
    # This must happen before model-specific preprocessing, as it expects a [0, 255] range image.
    if use_canny:
        image = apply_canny_tf(image, image_size[0], image_size[1])
    
    # 5. Apply the model-specific preprocess_function LAST.
    # This function scales pixel values to the range the model expects (e.g., [-1, 1] or [0, 1]).
    if preprocess_function:
        image = preprocess_function(image)

    return image, label


def create_balanced_dataset(class_0_ds, class_1_ds, num_class_0, num_class_1):
    """
    Balances the dataset by oversampling the minority class to a target ratio
    defined in the config file.
    
    Args:
        class_0_ds (tf.data.Dataset): Dataset for class 0.
        class_1_ds (tf.data.Dataset): Dataset for class 1.
        num_class_0 (int): Number of samples in class 0.
        num_class_1 (int): Number of samples in class 1.
        
    Returns:
        tf.data.Dataset: A balanced dataset.
    """
    minority_ratio = config.OVERSAMPLING_MINORITY_CLASS_RATIO
    
    # Determine the weights for sampling based on which class is the minority.
    # The weights correspond to the order of datasets: [class_0, class_1]
    if num_class_0 < num_class_1:
        # Class 0 is the minority
        sampling_weights = [minority_ratio, 1.0 - minority_ratio]
        print(f"✅ Balancing dataset: Class 0 is minority. Target ratio: {minority_ratio:.2f} (Class 0) / {1.0 - minority_ratio:.2f} (Class 1)")
    else:
        # Class 1 is the minority (or they are equal)
        sampling_weights = [1.0 - minority_ratio, minority_ratio]
        print(f"✅ Balancing dataset: Class 1 is minority. Target ratio: {1.0 - minority_ratio:.2f} (Class 0) / {minority_ratio:.2f} (Class 1)")

    # Repeat both datasets indefinitely
    class_0_ds = class_0_ds.repeat()
    class_1_ds = class_1_ds.repeat()
    
    # Use sample_from_datasets to create a balanced dataset with the desired ratio
    balanced_ds = tf.data.experimental.sample_from_datasets(
        [class_0_ds, class_1_ds],
        weights=sampling_weights
    )
    
    return balanced_ds


def create_dataset(data_dir, image_size, batch_size, preprocess_function, is_training=True, shuffle=True, use_canny=False):
    """
    Creates a tf.data.Dataset from image files in a directory.
    Handles class imbalance for the training set using random oversampling.

    Args:
        data_dir (str): Path to the data directory (e.g., 'data/train').
        image_size (tuple): The target (height, width) for the images.
        batch_size (int): The batch size.
        preprocess_function (callable): The model-specific preprocessing function.
        is_training (bool): If True, enables shuffling, augmentation, and oversampling.
        shuffle (bool): Whether to shuffle the dataset.
        use_canny (bool): If True, applies Canny edge detection.

    Returns:
        A tuple of (tf.data.Dataset, int, int) representing the dataset,
        number of samples in class 0, and number of samples in class 1.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    AUTOTUNE = tf.data.AUTOTUNE

    if is_training and config.USE_SMOTE:
        print("INFO: Using SMOTE to handle class imbalance. This may take a while as it loads all data into memory...")
        
        all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith('.jpg')]
        if not all_files:
            raise FileNotFoundError(f"No '.jpg' files found in directory: {data_dir}")

        X_train, y_train = [], []
        print(f"Loading {len(all_files)} images into memory for SMOTE preprocessing...")
        for file_path in tqdm(all_files):
            image = tf.io.read_file(file_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.cast(image, tf.float32)
            image = resize_and_pad_image(image, image_size, config.RESIZE_METHOD, config.USE_LETTERBOX_PADDING)
            if use_canny:
                image = apply_canny_tf(image, image_size[0], image_size[1])
            X_train.append(image.numpy())
            y_train.append(get_label_from_filename(file_path).numpy())
            
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        num_class_0, num_class_1 = np.bincount(y_train)
        print(f"  Loaded samples: {len(X_train)} (Class 0: {num_class_0}, Class 1: {num_class_1})")

        original_shape = X_train.shape
        X_train_flat = X_train.reshape(original_shape[0], -1)

        print("Applying SMOTE...")
        # Use the ratio from config to control SMOTE's aggressiveness
        smote = SMOTE(
            sampling_strategy=config.OVERSAMPLING_MINORITY_CLASS_RATIO,
            random_state=config.SEED
        )
        X_resampled_flat, y_resampled = smote.fit_resample(X_train_flat, y_train)
        
        new_shape = (X_resampled_flat.shape[0],) + original_shape[1:]
        X_resampled = X_resampled_flat.reshape(new_shape)
        
        num_class_0, num_class_1 = np.bincount(y_resampled)
        print(f"  SMOTE completed. New dataset size: {len(X_resampled)} (Class 0: {num_class_0}, Class 1: {num_class_1})")

        dataset = tf.data.Dataset.from_tensor_slices((X_resampled, y_resampled))
        
        if shuffle:
            dataset = dataset.shuffle(len(X_resampled))

        # Repeat the dataset indefinitely for training across multiple epochs
        dataset = dataset.repeat()

        def process_from_np(image, label):
            image = tf.cast(image, tf.float32)
            if is_training:
                image = data_augmentation(image, training=True)
            if preprocess_function:
                image = preprocess_function(image)
            return image, tf.cast(label, tf.int32)
            
        dataset = dataset.map(process_from_np, num_parallel_calls=AUTOTUNE)

    else:
        # List all image files and count them by class
        all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith('.jpg')]
        if not all_files:
            raise FileNotFoundError(f"No '.jpg' files found in directory: {data_dir}")

        class_0_files = [f for f in all_files if '_0.' in os.path.basename(f) or '_0.jpg' in os.path.basename(f)]
        class_1_files = [f for f in all_files if '_1.' in os.path.basename(f) or '_1.jpg' in os.path.basename(f)]
        num_class_0, num_class_1 = len(class_0_files), len(class_1_files)

        # Check for empty classes
        if is_training and (num_class_0 == 0 or num_class_1 == 0):
            raise ValueError(f"Training requires samples from both classes, but found "
                             f"Class 0: {num_class_0}, Class 1: {num_class_1} in {data_dir}")

        # Create a partial function for mapping, controlling augmentation with `is_training`
        process_func = lambda path, label: process_path(
            path, label, image_size, preprocess_function,
            apply_aug=is_training, use_canny=use_canny
        )

        if is_training: # Using standard oversampling
            print(f"Creating training dataset from: {data_dir}")
            print(f"  Training samples: {len(all_files)} (Class 0: {num_class_0}, Class 1: {num_class_1}) - Augmentation ENABLED")

            # Create separate datasets for each class
            ds_class_0 = tf.data.Dataset.from_tensor_slices(class_0_files)
            ds_class_1 = tf.data.Dataset.from_tensor_slices(class_1_files)

            # Shuffle each class dataset individually
            if shuffle:
                ds_class_0 = ds_class_0.shuffle(num_class_0)
                ds_class_1 = ds_class_1.shuffle(num_class_1)

            # Map the processing functions to get labels and process images
            labeled_ds_0 = ds_class_0.map(lambda x: (x, get_label_from_filename(x)), num_parallel_calls=AUTOTUNE)
            labeled_ds_1 = ds_class_1.map(lambda x: (x, get_label_from_filename(x)), num_parallel_calls=AUTOTUNE)
            
            # Apply the image processing
            processed_ds_0 = labeled_ds_0.map(process_func, num_parallel_calls=AUTOTUNE)
            processed_ds_1 = labeled_ds_1.map(process_func, num_parallel_calls=AUTOTUNE)

            # Create the final balanced dataset
            dataset = create_balanced_dataset(processed_ds_0, processed_ds_1, num_class_0, num_class_1)
            
        else: # For validation or testing
            print(f"Creating validation/test dataset from: {data_dir}")
            print(f"  Validation/Test samples: {len(all_files)} (Class 0: {num_class_0}, Class 1: {num_class_1}) - Augmentation DISABLED")
            
            # Create a single dataset for validation/testing
            list_ds = tf.data.Dataset.from_tensor_slices(all_files)
            if shuffle:
                list_ds = list_ds.shuffle(len(all_files))
                
            labeled_ds = list_ds.map(lambda x: (x, get_label_from_filename(x)), num_parallel_calls=AUTOTUNE)
            dataset = labeled_ds.map(process_func, num_parallel_calls=AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    
    return dataset, num_class_0, num_class_1