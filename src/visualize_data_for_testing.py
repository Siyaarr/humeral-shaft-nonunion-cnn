import argparse
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random

from . import config
from . import data_loader
from .config import MODEL_METADATA # Import the metadata from the training script

def denormalize_for_display(image):
    """
    Simple robust denormalization for display.
    Works for any preprocessing by scaling to [0,1] range.
    """
    image = image.numpy()
    
    # Simple min-max scaling to [0,1]
    img_min, img_max = image.min(), image.max()
    if img_max > img_min:
        image = (image - img_min) / (img_max - img_min)
    else:
        # If all values are the same, set to middle gray
        image = np.full_like(image, 0.5)
    
    return np.clip(image, 0, 1)

def visualize_data(model_name, source_dir_name, num_images=5, seed=42, show_raw_values=False):
    """
    Visualizes the effect of the complete preprocessing pipeline on a few sample images.
    Uses the EXACT same processing pipeline as training by calling data_loader.process_path.

    Args:
        model_name (str): The name of the model to get the preprocessing pipeline from.
        source_dir_name (str): The name of the data directory to use ('train', 'validation', or 'test').
        num_images (int): The number of sample images to display.
        seed (int): Random seed for consistent image selection.
        show_raw_values (bool): If True, shows the raw preprocessed values the model sees.
    """
    if model_name not in MODEL_METADATA:
        raise ValueError(f"Model '{model_name}' not defined in MODEL_METADATA.")
        
    # Set ALL random seeds for consistency
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # --- 1. Get Model-Specific Preprocessing Info ---
    metadata = MODEL_METADATA[model_name]
    image_size = metadata.get('required_image_size', config.DEFAULT_IMAGE_SIZE)
    preprocess_function = metadata['preprocess']
    
    print(f"--- Visualizing Preprocessing for Model: {model_name} ---")
    print(f"Target image size: {image_size}")
    print(f"Images source: {os.path.abspath(config.VALIDATION_DIR)} ({source_dir_name})")
    print(f"Random seed: {seed}")
    print(f"Show raw model input values: {show_raw_values}")
    print(f"Apply Canny Edge Detection from config: {config.USE_CANNY_EDGE}")

    # --- 2. Load Image Paths (with proper seeding) ---
    validation_dir = config.VALIDATION_DIR
    if not os.path.exists(validation_dir):
        print(f"Validation directory not found at '{validation_dir}'. Cannot visualize data.")
        return
        
    try:
        # Get all image paths and sort for consistency
        all_image_paths = []
        for file in os.listdir(validation_dir):
            if file.lower().endswith('.jpg'):
                all_image_paths.append(os.path.join(validation_dir, file))
        
        all_image_paths.sort()  # Sort for consistency
        
        if len(all_image_paths) < num_images:
            print(f"Only {len(all_image_paths)} JPG images found in '{validation_dir}'. Showing all available.")
            num_images = len(all_image_paths)
        
        if not all_image_paths:
            print(f"No JPG images found in '{validation_dir}'.")
            return
        
        # Use seed to select consistent images
        np.random.seed(seed)
        selected_indices = np.random.choice(len(all_image_paths), size=num_images, replace=False)
        selected_indices.sort()  # Keep in order for consistency
        image_paths = [all_image_paths[i] for i in selected_indices]
        
        print(f"Selected {len(image_paths)} images for visualization")
        
    except Exception as e:
        print(f"Error loading image paths: {e}")
        return

    # --- 3. Load Original Images ---
    original_images = []
    labels = []
    for path in image_paths:
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        original_images.append(img)
        
        # Extract label from filename for display
        filename = os.path.basename(path)
        if '_0.' in filename:
            labels.append('Healed (0)')
        elif '_1.' in filename:
            labels.append('Non-healed (1)')
        else:
            labels.append('Unknown')

    # --- 4. Apply Processing Using EXACT Same Pipeline as Training ---
    processed_no_aug = []
    processed_with_aug = []
    
    for i, path in enumerate(image_paths):
        # Get label for process_path (though we don't use it for visualization)
        if '_0.' in os.path.basename(path):
            label = 0
        elif '_1.' in os.path.basename(path):
            label = 1
        else:
            label = 0  # default
        
        # Process WITHOUT augmentation (exactly like validation set)
        processed_img_no_aug, _ = data_loader.process_path(
            path, label, image_size, preprocess_function, 
            apply_aug=False, use_canny=config.USE_CANNY_EDGE
        )
        processed_no_aug.append(processed_img_no_aug)
        
        # Process WITH augmentation (exactly like training set)
        processed_img_with_aug, _ = data_loader.process_path(
            path, label, image_size, preprocess_function, 
            apply_aug=True, use_canny=config.USE_CANNY_EDGE
        )
        processed_with_aug.append(processed_img_with_aug)

    # --- 5. Print Raw Value Statistics ---
    print("\n--- Raw Preprocessed Value Statistics (What the Model Actually Sees) ---")
    for i in range(min(2, num_images)):  # Show stats for first 2 images
        raw_values_no_aug = processed_no_aug[i].numpy()
        raw_values_aug = processed_with_aug[i].numpy()
        
        print(f"Image {i+1} (No Augmentation):")
        print(f"  Value range: [{raw_values_no_aug.min():.3f}, {raw_values_no_aug.max():.3f}]")
        print(f"  Mean: {raw_values_no_aug.mean():.3f}, Std: {raw_values_no_aug.std():.3f}")
        
        print(f"Image {i+1} (With Augmentation):")
        print(f"  Value range: [{raw_values_aug.min():.3f}, {raw_values_aug.max():.3f}]")
        print(f"  Mean: {raw_values_aug.mean():.3f}, Std: {raw_values_aug.std():.3f}")
        print()

    # --- 6. Display the Images ---
    if show_raw_values:
        # Show 4 columns: Original, Processed (display), Processed (raw), Augmented (raw)
        fig, axes = plt.subplots(num_images, 4, figsize=(20, 5 * num_images))
        if num_images == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Data Visualization for {model_name} (Including Raw Values)\nImages from: {os.path.basename(config.VALIDATION_DIR)}\nSeed: {seed}', fontsize=14)
        
        for i in range(num_images):
            # Original Image
            axes[i, 0].imshow(original_images[i].numpy().astype("uint8"))
            axes[i, 0].set_title(f"Original Image {i+1}\nLabel: {labels[i]}")
            axes[i, 0].axis("off")

            # Processed (Display-friendly)
            processed_img_no_aug = denormalize_for_display(processed_no_aug[i])
            axes[i, 1].imshow(processed_img_no_aug)
            axes[i, 1].set_title(f"Processed (Display)\nSize: {image_size}")
            axes[i, 1].axis("off")
            
            # Processed (Raw values the model sees)
            raw_img = processed_no_aug[i].numpy()
            # Normalize raw values for display while showing the actual range
            raw_img_display = denormalize_for_display(processed_no_aug[i])
            axes[i, 2].imshow(raw_img_display, cmap='gray')
            axes[i, 2].set_title(f"Raw Model Input\nRange: [{raw_img.min():.2f}, {raw_img.max():.2f}]")
            axes[i, 2].axis("off")
            
            # Augmented (Raw values)
            raw_aug_img = processed_with_aug[i].numpy()
            # Normalize raw augmented values for display while showing the actual range
            raw_aug_img_display = denormalize_for_display(processed_with_aug[i])
            axes[i, 3].imshow(raw_aug_img_display, cmap='gray')
            axes[i, 3].set_title(f"Raw Augmented Input\nRange: [{raw_aug_img.min():.2f}, {raw_aug_img.max():.2f}]")
            axes[i, 3].axis("off")
    else:
        # Standard 3-column display
        fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
        if num_images == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Data Visualization for {model_name}\nImages from: {os.path.basename(config.VALIDATION_DIR)}\nSeed: {seed}', fontsize=14)
        
        for i in range(num_images):
            # --- Display Original Image ---
            axes[i, 0].imshow(original_images[i].numpy().astype("uint8"))
            axes[i, 0].set_title(f"Original Image {i+1}\nLabel: {labels[i]}")
            axes[i, 0].axis("off")

            # --- Display Processed Image (No Augmentation) ---
            processed_img_no_aug = denormalize_for_display(processed_no_aug[i])
            axes[i, 1].imshow(processed_img_no_aug)
            axes[i, 1].set_title(f"Processed (No Aug)\nSize: {image_size}")
            axes[i, 1].axis("off")
            
            # --- Display Processed Image (With Augmentation) ---
            processed_img_aug = denormalize_for_display(processed_with_aug[i])
            axes[i, 2].imshow(processed_img_aug)
            axes[i, 2].set_title(f"Processed + Augmented\nSize: {image_size}")
            axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize the data preprocessing pipeline for a given model.")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help=f"The name of the model to visualize. Available models: {list(MODEL_METADATA.keys())}"
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help="The data source to visualize."
    )
    parser.add_argument(
        '--num_images',
        type=int,
        default=5,
        help="Number of sample images to display."
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed for consistent image selection."
    )
    parser.add_argument(
        '--show_raw_values',
        action='store_true',
        help="Show the raw preprocessed values that the model actually sees."
    )
    
    # Example usage:
    # python -m src.visualize_data_for_testing --model EfficientNetB0 --num_images 3 --seed 42 --source validation
    # python -m src.visualize_data_for_testing --model DenseNet201 --show_raw_values --source validation
    
    parsed_args = parser.parse_args()
    visualize_data(
        parsed_args.model, 
        parsed_args.source, 
        parsed_args.num_images, 
        parsed_args.seed, 
        parsed_args.show_raw_values
    ) 