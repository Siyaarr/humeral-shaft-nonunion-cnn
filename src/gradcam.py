import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import argparse
import json
from . import config
from . import data_loader

class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) implementation
    for visualizing what CNNs focus on when making predictions.
    """
    
    def __init__(self, model_path, layer_name=None):
        """
        Initialize Grad-CAM with a trained model.
        
        Args:
            model_path (str): Path to the saved Keras model
            layer_name (str): Name of the convolutional layer to visualize
                             If None, will use the last convolutional layer
        """
        self.model = tf.keras.models.load_model(model_path)
        self.layer_name = layer_name
        self.grad_model = None
        
        # If no layer specified, find the last conv layer
        if self.layer_name is None:
            self.layer_name = self._find_last_conv_layer()
        
        self._build_grad_model()
        
    def _find_last_conv_layer(self):
        """Find the last convolutional layer in the base model."""
        base_model = self.model.layers[1]  # Base model is typically the second layer
        
        conv_layers = []
        for layer in base_model.layers:
            if 'conv' in layer.name.lower() or isinstance(layer, tf.keras.layers.Conv2D):
                conv_layers.append(layer.name)
        
        if conv_layers:
            last_conv = conv_layers[-1]
            print(f"Using last convolutional layer: {last_conv}")
            return last_conv
        else:
            raise ValueError("No convolutional layers found in the model!")
    
    def _build_grad_model(self):
        """Build the gradient model for Grad-CAM computation."""
        base_model = self.model.layers[1]
        
        # Find the target layer
        target_layer = None
        for layer in base_model.layers:
            if layer.name == self.layer_name:
                target_layer = layer
                break
        
        if target_layer is None:
            raise ValueError(f"Layer '{self.layer_name}' not found in the model!")
        
        # Store references for gradient computation
        self.base_model = base_model
        self.target_layer = target_layer
        self.grad_model = None  # We'll compute gradients manually
        
        print(f"Grad-CAM setup for layer: {self.layer_name}")
        print(f"Target layer output shape: {target_layer.output_shape}")
    
    def generate_heatmap(self, image, class_idx=None, eps=1e-8):
        """
        Generate Grad-CAM heatmap for an input image.
        
        Args:
            image (tf.Tensor): Input image tensor (1, H, W, C)
            class_idx (int): Class index to generate heatmap for
                           If None, uses the predicted class
            eps (float): Small epsilon to avoid division by zero
            
        Returns:
            numpy.ndarray: Heatmap of shape (H, W)
        """
        # Ensure image is in the right format
        if len(image.shape) == 3:
            image = tf.expand_dims(image, axis=0)
        
        try:
            # Get the base model (VGG, ResNet, etc.)
            base_model = self.model.layers[1]  # Skip input layer
            
            # Create a model that maps the input to the target conv layer output
            conv_layer = base_model.get_layer(self.layer_name)
            feature_extractor = tf.keras.Model(
                inputs=base_model.input,
                outputs=conv_layer.output
            )
            
            # Create a model that maps conv features to final prediction
            # We need to trace from conv layer through the rest of the model
            conv_output_shape = conv_layer.output_shape
            
            # Create input for the classifier part
            classifier_input = tf.keras.Input(shape=conv_output_shape[1:])
            
            # Find all layers after the target conv layer
            layers_after_conv = []
            found_target = False
            for layer in base_model.layers:
                if found_target:
                    layers_after_conv.append(layer)
                elif layer.name == self.layer_name:
                    found_target = True
            
            # Build the classifier path
            x = classifier_input
            for layer in layers_after_conv:
                x = layer(x)
            
            # Add the final classification layers from the full model
            # These are the layers after the base model
            for layer in self.model.layers[2:]:  # Skip input and base model
                x = layer(x)
            
            classifier_model = tf.keras.Model(classifier_input, x)
            
            # Now we can compute gradients properly
            with tf.GradientTape() as tape:
                # Get conv features
                conv_outputs = feature_extractor(image)
                tape.watch(conv_outputs)
                
                # Get predictions from conv features
                predictions = classifier_model(conv_outputs)
                
                # Determine which class to explain
                if class_idx is None:
                    if predictions.shape[-1] == 1:
                        # Binary classification - use the probability score
                        class_score = predictions[0, 0]
                    else:
                        # Multi-class - use max prediction
                        class_idx = tf.argmax(predictions[0])
                        class_score = predictions[0, class_idx]
                else:
                    if predictions.shape[-1] == 1:
                        # Binary classification
                        class_score = predictions[0, 0] if class_idx == 1 else 1 - predictions[0, 0]
                    else:
                        # Multi-class
                        class_score = predictions[0, class_idx]
            
            # Compute gradients
            grads = tape.gradient(class_score, conv_outputs)
            
            if grads is None:
                raise ValueError("Gradients returned None - model graph connectivity issue")
            
            # Guided backpropagation - keep only positive gradients
            grads = tf.cast(grads > 0, tf.float32) * grads
            
            # Global average pooling of gradients (importance weights)
            pooled_grads = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
            
            # Weight the feature maps by their importance
            conv_outputs_weighted = conv_outputs * pooled_grads
            
            # Sum across all feature maps
            heatmap = tf.reduce_sum(conv_outputs_weighted, axis=-1)
            heatmap = tf.squeeze(heatmap)
            
            # Apply ReLU to keep only positive activations
            heatmap = tf.maximum(heatmap, 0)
            
            # Normalize the heatmap to [0, 1]
            heatmap_max = tf.reduce_max(heatmap)
            if heatmap_max > eps:
                heatmap = heatmap / heatmap_max
            
            return heatmap.numpy()
            
        except Exception as e:
            print(f"Error in generate_heatmap: {e}")
            # Fallback: return a simple activation map without gradients
            try:
                base_model = self.model.layers[1]
                conv_layer = base_model.get_layer(self.layer_name)
                feature_extractor = tf.keras.Model(
                    inputs=base_model.input,
                    outputs=conv_layer.output
                )
                
                conv_outputs = feature_extractor(image)
                # Simple average of all feature maps
                heatmap = tf.reduce_mean(conv_outputs, axis=-1)
                heatmap = tf.squeeze(heatmap)
                heatmap = tf.maximum(heatmap, 0)
                
                heatmap_max = tf.reduce_max(heatmap)
                if heatmap_max > eps:
                    heatmap = heatmap / heatmap_max
                
                print("Using fallback activation map (no gradients)")
                return heatmap.numpy()
            except:
                print("Both gradient and fallback methods failed")
                return None
    
    def visualize_gradcam(self, image, heatmap, alpha=0.6, colormap=cm.jet):
        """
        Create a visualization overlaying the heatmap on the original image.
        
        Args:
            image (numpy.ndarray): Original image (H, W, C)
            heatmap (numpy.ndarray): Grad-CAM heatmap (H, W)
            alpha (float): Transparency for the heatmap overlay
            colormap: Matplotlib colormap for the heatmap
            
        Returns:
            numpy.ndarray: Visualization image
        """
        # Normalize image to [0, 1]
        if image.max() > 1:
            image = image / 255.0
        
        # Resize heatmap to match image dimensions
        heatmap_resized = tf.image.resize(
            heatmap[..., tf.newaxis], 
            [image.shape[0], image.shape[1]]
        )
        heatmap_resized = tf.squeeze(heatmap_resized).numpy()
        
        # Apply colormap to heatmap
        heatmap_colored = colormap(heatmap_resized)
        heatmap_colored = heatmap_colored[:, :, :3]  # Remove alpha channel
        
        # Overlay heatmap on image
        visualization = (1 - alpha) * image + alpha * heatmap_colored
        
        return visualization
    
    def explain_prediction(self, image_path, save_dir, class_names=None, preprocess_func=None):
        """
        Generate complete Grad-CAM explanation for a single image.
        
        Args:
            image_path (str): Path to the input image
            save_dir (str): Directory to save visualizations
            class_names (list): List of class names for labeling
            preprocess_func: Preprocessing function to apply to the image
            
        Returns:
            dict: Dictionary containing prediction info and file paths
        """
        if class_names is None:
            class_names = ['Healed', 'Non-healed']
        
        # Load and preprocess image
        try:
            # Load image
            image = tf.keras.preprocessing.image.load_img(image_path)
            image_array = tf.keras.preprocessing.image.img_to_array(image)
            
            # Get model requirements
            target_size = self.model.input_shape[1:3]  # (H, W)
            
            # Resize image to model requirements
            image_resized = tf.image.resize(image_array, target_size)
            
            # Apply model-specific preprocessing if provided
            if preprocess_func is not None:
                image_preprocessed = preprocess_func(tf.expand_dims(image_resized, axis=0))
            else:
                # Default preprocessing: normalize to [0, 1]
                image_preprocessed = tf.expand_dims(image_resized / 255.0, axis=0)
            
            # Make prediction
            predictions = self.model(image_preprocessed)
            predicted_class = tf.argmax(predictions[0]).numpy()
            confidence = float(predictions[0][predicted_class])
            
            print(f"Prediction: {class_names[predicted_class]} ({confidence:.4f})")
            
            # Generate heatmap
            heatmap = self.generate_heatmap(image_preprocessed)
            
            if heatmap is None:
                print("❌ Failed to generate heatmap")
                return None
            
            # Create visualizations
            image_np = image_resized.numpy() / 255.0
            visualization = self.visualize_gradcam(image_np, heatmap)
            
            # Create figure with multiple subplots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(image_np)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Heatmap only
            im1 = axes[1].imshow(heatmap, cmap='jet')
            axes[1].set_title('Grad-CAM Heatmap')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            
            # Overlay
            axes[2].imshow(visualization)
            axes[2].set_title(f'Grad-CAM Overlay\n{class_names[predicted_class]} ({confidence:.3f})')
            axes[2].axis('off')
            
            plt.suptitle(f'Grad-CAM Analysis - Layer: {self.layer_name}', fontsize=14)
            plt.tight_layout()
            
            # Save visualization
            os.makedirs(save_dir, exist_ok=True)
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            save_path = os.path.join(save_dir, f'gradcam_{image_name}_{self.layer_name}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            result = {
                'image_path': image_path,
                'prediction': class_names[predicted_class],
                'confidence': confidence,
                'predicted_class_idx': int(predicted_class),
                'layer_name': self.layer_name,
                'save_path': save_path
            }
            
            print(f"✅ Grad-CAM analysis saved to: {save_path}")
            return result
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            return None
    
    def batch_analysis(self, image_dir, save_dir, max_images=10, class_names=None, preprocess_func=None):
        """
        Perform Grad-CAM analysis on multiple images from a directory.
        
        Args:
            image_dir (str): Directory containing images
            save_dir (str): Directory to save visualizations
            max_images (int): Maximum number of images to process
            class_names (list): List of class names
            preprocess_func: Preprocessing function to apply to images
            
        Returns:
            list: List of analysis results
        """
        if class_names is None:
            class_names = ['Healed', 'Non-healed']
        
        # Get image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = []
        
        for ext in image_extensions:
            image_files.extend([f for f in os.listdir(image_dir) if f.lower().endswith(ext)])
        
        image_files = image_files[:max_images]
        
        if not image_files:
            print(f"No image files found in {image_dir}")
            return []
        
        print(f"Processing {len(image_files)} images...")
        
        results = []
        for i, image_file in enumerate(image_files, 1):
            print(f"\n--- Processing image {i}/{len(image_files)}: {image_file} ---")
            image_path = os.path.join(image_dir, image_file)
            result = self.explain_prediction(image_path, save_dir, class_names)
            if result:
                results.append(result)
        
        # Create summary report
        summary_path = os.path.join(save_dir, 'gradcam_analysis_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\n🎉 Batch analysis complete!")
        print(f"📁 Results saved to: {save_dir}")
        print(f"📊 Summary report: {summary_path}")
        print(f"✅ Processed {len(results)} images successfully!")
        
        return results

def run_gradcam_analysis(model_name, run_id, image_source, layer_name=None, max_images=10):
    """
    Main function to run Grad-CAM analysis.
    
    Args:
        model_name (str): Name of the model
        run_id (str): ID of the specific run
        image_source (str): Path to image file or directory
        layer_name (str): Specific layer name (optional)
        max_images (int): Maximum images to process if directory
    """
    print(f"\n=== Grad-CAM Analysis for {model_name} ({run_id}) ===")
    
    # Validate model name
    if model_name not in config.MODEL_METADATA:
        print(f"❌ Model '{model_name}' not found in config!")
        return
    
    # Get preprocessing function for this model
    preprocess_func = config.MODEL_METADATA[model_name]['preprocess']
    print(f"Using preprocessing: {preprocess_func.__name__}")
    
    # Check if model exists
    model_path = os.path.join('results', model_name, run_id, config.MODEL_CHECKPOINT_FILENAME)
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return
    
    # Create output directory
    output_dir = os.path.join('results', model_name, run_id, 'gradcam_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize Grad-CAM
        gradcam = GradCAM(model_path, layer_name)
        
        # Check if image_source is file or directory
        if os.path.isfile(image_source):
            print(f"Analyzing single image: {image_source}")
            result = gradcam.explain_prediction(image_source, output_dir, preprocess_func=preprocess_func)
            if result:
                print(f"✅ Analysis complete! Check: {result['save_path']}")
        elif os.path.isdir(image_source):
            print(f"Analyzing images in directory: {image_source}")
            results = gradcam.batch_analysis(image_source, output_dir, max_images, preprocess_func=preprocess_func)
            print(f"✅ Processed {len(results)} images successfully!")
        else:
            print(f"❌ Image source not found: {image_source}")
            return
            
    except Exception as e:
        print(f"❌ Error during Grad-CAM analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Grad-CAM analysis for trained models")
    parser.add_argument('--model', type=str, required=True,
                       help='Model name (e.g., VGG16)')
    parser.add_argument('--run', type=str, required=True,
                       help='Run ID (e.g., run_1)')
    parser.add_argument('--images', type=str, required=True,
                       help='Path to image file or directory')
    parser.add_argument('--layer', type=str, default=None,
                       help='Specific layer name for Grad-CAM (optional)')
    parser.add_argument('--max-images', type=int, default=10,
                       help='Maximum number of images to process (for directories)')
    
    args = parser.parse_args()
    
    # Example usage:
    # python -m src.gradcam --model VGG16 --run run_1 --images path/to/validation/images
    # python -m src.gradcam --model VGG16 --run run_1 --images path/to/single/image.jpg --layer block5_conv3
    
    run_gradcam_analysis(args.model, args.run, args.images, args.layer, args.max_images) 