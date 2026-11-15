import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from . import config

def build_model(base_model_loader, input_shape, dense_units, dropout_rate, kernel_regularizer_l2, use_dense_layer):
    """
    Builds a classification model with a specified base model and a universal head.

    This function creates a model by loading a pre-trained base model, freezing its
    layers, and adding a classification head. The head's architecture is adaptive:
    it adds a GlobalAveragePooling2D layer only if the base model's output is
    4D (like a CNN's feature map), making it compatible with both CNNs and
    Transformer-based models (like ViT) that output 2D feature vectors.

    Args:
        base_model_loader (callable): A function that returns an instantiated
                                      Keras model (e.g., tf.keras.applications.VGG16).
        input_shape (tuple): The shape of the input images (height, width, channels).
        dense_units (int): The number of units in the fixed dense layer of the head.
        dropout_rate (float): The dropout rate to apply in the head.
        kernel_regularizer_l2 (float): The L2 regularization factor for the dense layer.
        use_dense_layer (bool): If True, includes the intermediate dense layer.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    # Load the base model using the provided loader function.
    # `include_top=False` removes the original classification layer.
    base_model = base_model_loader(include_top=False,
                                   weights='imagenet',
                                   input_shape=input_shape)

    # Freeze the layers of the base model to prevent them from being updated
    # during the initial feature extraction training phase.
    base_model.trainable = False

    # Define the model input
    inputs = layers.Input(shape=input_shape)
    
    # Pass inputs through the base model.
    # By not specifying `training`, we let Keras handle it automatically.
    # It will be True during `fit()` and False during `evaluate()` or `predict()`.
    x = base_model(inputs)

    # --- Intelligent and Universal Head Architecture ---
    # Inspect the output of the base model to adapt the head architecture.
    # If the output tensor is 4-dimensional (batch, height, width, channels),
    # it's a feature map from a CNN. Apply Global Average Pooling to flatten it.
    if len(x.shape) == 4:
        x = layers.GlobalAveragePooling2D()(x)
    # If the output is 2-dimensional (batch, features), it's likely from a
    # Transformer model (e.g., ViT, DeiT) which already provides a 1D feature
    # vector (like a CLS token). No pooling is needed.

    # --- Standard Head ---
    # Add the classification head, which is consistent for all models.
    # The intermediate dense layer is optional, controlled by the `use_dense_layer` flag.
    if use_dense_layer:
        print("  - Head Configuration: Including intermediate Dense layer.")
        x = layers.Dense(
            units=dense_units,
            activation='relu',
            kernel_regularizer=regularizers.l2(kernel_regularizer_l2)
        )(x)
    else:
        print("  - Head Configuration: Skipping intermediate Dense layer (GAP > Dropout > Output).")

    # A dropout layer is always included for regularization.
    x = layers.Dropout(rate=dropout_rate)(x)
    
    # The final output layer for binary classification.
    # The activation is 'sigmoid' to produce a probability between 0 and 1.
    outputs = layers.Dense(config.NUM_CLASSES, activation='sigmoid')(x)

    # Create the final model
    model = models.Model(inputs, outputs)
    
    print(f"Model built successfully. Base model: {base_model.name}")
    print(f"  - Base model output shape: {base_model.output_shape}")
    if len(base_model.output_shape) == 4:
        print("  - Adaptive Head: GlobalAveragePooling2D was ADDED.")
    else:
        print("  - Adaptive Head: GlobalAveragePooling2D was SKIPPED (2D output detected).")

    return model

def unfreeze_layers(model):
    """
    Unfreezes the top layers of the base model for fine-tuning.

    This function makes the base model trainable and then unfreezes a percentage
    of its layers, determined by `config.FINE_TUNE_LAYER_PERCENTAGE`. The layers
    at the end of the model (the "top" layers) are unfrozen, as they typically
    learn more specialized features that are more suitable for tuning.

    Args:
        model (tf.keras.Model): The model to be fine-tuned.

    Returns:
        None: The model is modified in place.
    """
    # The base model is the first layer of our complete model.
    base_model = model.layers[1]
    base_model.trainable = True

    # Calculate the number of layers to unfreeze
    num_layers = len(base_model.layers)
    fine_tune_at = int(num_layers * (1 - config.FINE_TUNE_LAYER_PERCENTAGE))

    print(f"\nUnfreezing model for fine-tuning...")
    print(f"  - Base model: {base_model.name} has {num_layers} layers.")
    print(f"  - Unfreezing top {config.FINE_TUNE_LAYER_PERCENTAGE * 100:.1f}% of layers.")
    print(f"  - Layers from index {fine_tune_at} onwards will be trainable.")

    # Freeze all layers up to the calculated index
    for i, layer in enumerate(base_model.layers[:fine_tune_at]):
        layer.trainable = False

    # The rest of the layers (the top ones) will be trainable
    for i, layer in enumerate(base_model.layers[fine_tune_at:]):
        layer.trainable = True 