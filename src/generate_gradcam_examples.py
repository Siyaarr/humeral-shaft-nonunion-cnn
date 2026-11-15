import os
import re
import json
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from . import config
from . import model as model_builder
from . import data_loader


def read_final_model_details(final_folder):
    details_path = os.path.join(final_folder, 'final_model_details.json')
    if not os.path.exists(details_path):
        raise FileNotFoundError(f"final_model_details.json not found in {final_folder}")
    with open(details_path, 'r') as f:
        details = json.load(f)
    return details


def build_and_load_final_model(final_folder):
    details = read_final_model_details(final_folder)
    model_name = details['model_name']
    hyperparameters = details['hyperparameters']

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

    # Match fine-tuning state, compile, then load weights
    model_builder.unfreeze_layers(model)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )

    weights_path = os.path.join(final_folder, 'final_fine_tuned_model_weights.h5')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    model.load_weights(weights_path)

    return model, model_name, hyperparameters, image_size, metadata['preprocess']


def parse_label_from_filename(path):
    name = os.path.splitext(os.path.basename(path))[0]
    if re.search(r'_0$', name):
        return 0
    if re.search(r'_1$', name):
        return 1
    # fallback: last underscore part
    parts = name.split('_')
    try:
        return int(parts[-1])
    except Exception:
        raise ValueError(f"Cannot parse label from filename: {path}")


def load_and_preprocess(path, image_size, preprocess_func, use_canny):
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.cast(image, tf.float32)
    image_resized = data_loader.resize_and_pad_image(
        image, image_size, config.RESIZE_METHOD, config.USE_LETTERBOX_PADDING
    )
    if use_canny:
        image_resized = data_loader.apply_canny_tf(image_resized, image_size[0], image_size[1])
    image_for_overlay = image_resized / 255.0
    image_preprocessed = preprocess_func(image_resized) if preprocess_func else image_for_overlay
    image_preprocessed = tf.expand_dims(image_preprocessed, axis=0)
    return image_for_overlay.numpy(), image_preprocessed


def find_last_conv_layer_in_base(model):
    base_model = model.layers[1]
    last_name = None
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) or 'conv' in layer.name.lower():
            last_name = layer.name
    if not last_name:
        raise ValueError("No convolutional layer found in base model")
    return last_name


def build_feature_and_classifier_models(model, target_layer_name):
    base_model = model.layers[1]
    target_layer = base_model.get_layer(target_layer_name)

    feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=target_layer.output)

    conv_output_shape = target_layer.output_shape
    classifier_input = tf.keras.Input(shape=conv_output_shape[1:])

    # Layers after target conv inside base model
    x = classifier_input
    found = False
    for layer in base_model.layers:
        if found:
            x = layer(x)
        elif layer.name == target_layer_name:
            found = True

    # Then the head layers after base model in the full model
    for layer in model.layers[2:]:
        x = layer(x)

    classifier_model = tf.keras.Model(classifier_input, x)
    return feature_extractor, classifier_model


def compute_gradcam_heatmap(
    model,
    image_preprocessed,
    target_layer_name,
    eps=1e-8,
    cam_class='pred',  # 'pos' | 'pred' | 'true'
    y_true=None,
    pred_prob=None,
    pred_threshold=0.5,
):
    feature_extractor, classifier_model = build_feature_and_classifier_models(model, target_layer_name)

    with tf.GradientTape() as tape:
        conv_outputs = feature_extractor(image_preprocessed)
        tape.watch(conv_outputs)
        predictions = classifier_model(conv_outputs)
        p = predictions[0, 0]
        # Choose target class for Grad-CAM
        if cam_class == 'pos':
            class_score = p
        elif cam_class == 'true':
            if y_true is None:
                class_score = p
            else:
                class_score = p if int(y_true) == 1 else 1.0 - p
        else:  # 'pred'
            if pred_prob is None:
                class_score = p
            else:
                class_score = p if float(pred_prob) >= float(pred_threshold) else 1.0 - p

    grads = tape.gradient(class_score, conv_outputs)
    if grads is None:
        raise ValueError("Gradients are None; check model graph connectivity")

    grads = tf.cast(grads > 0, tf.float32) * grads
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    conv_outputs_weighted = conv_outputs * pooled_grads
    heatmap = tf.reduce_sum(conv_outputs_weighted, axis=-1)
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > eps:
        heatmap = heatmap / max_val
    return heatmap.numpy()


def overlay_heatmap_on_image(
    image_np,
    heatmap,
    alpha=0.25,
    blend_mode='screen',  # 'linear' or 'screen'
    use_mask=True,
    threshold=0.4,
    gamma=0.9,
    bg_contrast='percentile',  # 'none' | 'minmax' | 'percentile'
    bg_q_low=2.0,
    bg_q_high=98.0,
):
    if image_np.max() > 1:
        image_np = image_np / 255.0
    h, w = image_np.shape[:2]
    heatmap_resized = tf.image.resize(heatmap[..., tf.newaxis], [h, w])
    heatmap_resized = tf.squeeze(heatmap_resized).numpy()
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]

    # Enhance background contrast for display (does not affect Grad-CAM)
    if bg_contrast != 'none':
        img = image_np.copy()
        if bg_contrast == 'minmax':
            vmin = float(np.min(img))
            vmax = float(np.max(img))
        else:  # percentile
            # Ignore pure zeros (letterbox) when computing percentiles
            mask = img > 0
            if np.any(mask):
                vmin = float(np.percentile(img[mask], bg_q_low))
                vmax = float(np.percentile(img[mask], bg_q_high))
            else:
                vmin = float(np.min(img))
                vmax = float(np.max(img))
        denom = max(1e-6, vmax - vmin)
        image_np = np.clip((img - vmin) / denom, 0.0, 1.0)

    if use_mask:
        heat_clipped = np.clip(heatmap_resized, 0.0, 1.0)
        safe_denom = max(1e-6, 1.0 - threshold)
        ramp = np.clip((heat_clipped - threshold) / safe_denom, 0.0, 1.0)
        alpha_map = alpha * np.power(ramp, gamma)
        alpha_map = np.expand_dims(alpha_map, axis=-1)
    else:
        alpha_map = alpha

    if blend_mode == 'linear':
        overlay = (1 - alpha_map) * image_np + alpha_map * heatmap_colored
    else:  # screen blend for brighter background: 1 - (1-a)*(1-b)
        # Modulate heatmap contribution by alpha_map
        contrib = heatmap_colored * alpha_map
        overlay = 1.0 - (1.0 - image_np) * (1.0 - contrib)

    overlay = np.clip(overlay, 0.0, 1.0)
    return overlay


def choose_threshold(pred_probs, y_true):
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, pred_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    if thresholds.size == 0:
        return 0.5
    idx = np.argmax(f1_scores[:-1])
    return float(thresholds[idx])


def collect_predictions(model, test_dir, image_size, preprocess_func, use_canny):
    image_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith('.jpg')]
    records = []
    for path in image_paths:
        y_true = parse_label_from_filename(path)
        img_overlay, img_pre = load_and_preprocess(path, image_size, preprocess_func, use_canny)
        prob = float(model.predict(img_pre, verbose=0).flatten()[0])
        records.append({'path': path, 'y_true': y_true, 'prob': prob, 'overlay_img': img_overlay})
    return records


def select_examples_by_group(records, threshold, per_group):
    groups = {'TP': [], 'TN': [], 'FP': [], 'FN': []}
    for r in records:
        pred = 1 if r['prob'] > threshold else 0
        if pred == 1 and r['y_true'] == 1:
            groups['TP'].append(r)
        elif pred == 0 and r['y_true'] == 0:
            groups['TN'].append(r)
        elif pred == 1 and r['y_true'] == 0:
            groups['FP'].append(r)
        elif pred == 0 and r['y_true'] == 1:
            groups['FN'].append(r)
    # Trim to requested count
    for k in groups:
        groups[k] = groups[k][:per_group]
    return groups


def save_group_gradcams(
    model,
    groups,
    image_size,
    preprocess_func,
    use_canny,
    target_layer_name,
    out_dir,
    overlay_alpha=0.25,
    blend_mode='screen',
    use_mask=True,
    heatmap_threshold=0.4,
    heatmap_gamma=0.9,
    overlay_use_canny=False,
    bg_contrast='percentile',
    bg_q_low=2.0,
    bg_q_high=98.0,
    cam_class='pred',
    cam_threshold=0.5,
):
    os.makedirs(out_dir, exist_ok=True)
    for group_name, items in groups.items():
        group_dir = os.path.join(out_dir, group_name)
        os.makedirs(group_dir, exist_ok=True)
        for idx, r in enumerate(items):
            # Preprocess again to ensure correct tensor for Grad-CAM (keeps code simple and robust)
            # For visualization background, avoid model preprocessing and (by default) avoid Canny
            img_overlay, _ = load_and_preprocess(r['path'], image_size, None, overlay_use_canny)
            # For Grad-CAM computation, use the exact model preprocessing and canny setting
            _, img_pre = load_and_preprocess(r['path'], image_size, preprocess_func, use_canny)
            heatmap = compute_gradcam_heatmap(
                model,
                img_pre,
                target_layer_name,
                cam_class=cam_class,
                y_true=r['y_true'],
                pred_prob=r['prob'],
                pred_threshold=cam_threshold,
            )
            vis = overlay_heatmap_on_image(
                img_overlay,
                heatmap,
                alpha=overlay_alpha,
                blend_mode=blend_mode,
                use_mask=use_mask,
                threshold=heatmap_threshold,
                gamma=heatmap_gamma,
                bg_contrast=bg_contrast,
                bg_q_low=bg_q_low,
                bg_q_high=bg_q_high,
            )

            base = os.path.splitext(os.path.basename(r['path']))[0]
            fname = f"{group_name}_{idx+1}_true{r['y_true']}_prob{r['prob']:.3f}_{base}.png"
            save_path = os.path.join(group_dir, fname)
            plt.figure(figsize=(6, 6))
            plt.imshow(vis)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate Grad-CAM examples by confusion groups (TP/TN/FP/FN).')
    parser.add_argument('--final-model-folder', type=str, default=os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'results - FINAL', 'final_model')),
                        help='Folder containing final_model_details.json and final_fine_tuned_model_weights.h5')
    parser.add_argument('--per-group', type=int, default=4, help='Number of images per group to generate (default: 4)')
    parser.add_argument('--layer', type=str, default=None, help='Specific conv layer for Grad-CAM (optional)')
    parser.add_argument('--threshold', type=float, default=None, help='Decision threshold; if omitted, auto-optimized by PR F1')
    parser.add_argument('--overlay-alpha', type=float, default=0.25, help='Heatmap transparency over original image (lower shows more background)')
    parser.add_argument('--blend-mode', type=str, choices=['linear', 'screen'], default='screen', help='Overlay blend mode (screen keeps background bright)')
    parser.add_argument('--heatmap-threshold', type=float, default=0.4, help='Only emphasize heatmap where normalized value >= threshold (0..1)')
    parser.add_argument('--heatmap-gamma', type=float, default=0.9, help='Gamma for heatmap intensity ramping (higher emphasizes peaks)')
    parser.add_argument('--masking-off', action='store_true', help='Disable masking (overlay everywhere)')
    parser.add_argument('--overlay-use-canny', action='store_true', help='Use Canny image as overlay background as well (default: off)')
    parser.add_argument('--bg-contrast', type=str, choices=['none', 'minmax', 'percentile'], default='percentile', help='Contrast normalization for background image')
    parser.add_argument('--bg-q-low', type=float, default=2.0, help='Low percentile for background contrast')
    parser.add_argument('--bg-q-high', type=float, default=98.0, help='High percentile for background contrast')
    parser.add_argument('--cam-class', type=str, choices=['pos', 'pred', 'true'], default='pred', help='Class target for Grad-CAM (pos=positive, pred=predicted, true=true label)')
    parser.add_argument('--cam-threshold', type=float, default=0.5, help='Prediction threshold for cam-class=pred')
    parser.add_argument('--out-dir', type=str, default=None, help='Output folder name inside the final-model folder (default: gradcam_examples)')
    args = parser.parse_args()

    final_folder = args.final_model_folder
    model, model_name, hps, image_size, preprocess_func = build_and_load_final_model(final_folder)
    use_canny = hps.get('use_canny', False)

    # Collect predictions over test set
    test_dir = config.TEST_DIR
    records = collect_predictions(model, test_dir, image_size, preprocess_func, use_canny)

    # Choose threshold
    if args.threshold is not None:
        threshold = float(args.threshold)
    else:
        # Try reading from saved final test results first
        saved_metrics_path = os.path.join(final_folder, 'final_test_results.json')
        if os.path.exists(saved_metrics_path):
            try:
                with open(saved_metrics_path, 'r') as f:
                    saved = json.load(f)
                threshold = float(saved['final_test_metrics']['optimal_threshold'])
            except Exception:
                threshold = choose_threshold(
                    np.array([r['prob'] for r in records]),
                    np.array([r['y_true'] for r in records])
                )
        else:
            threshold = choose_threshold(
                np.array([r['prob'] for r in records]),
                np.array([r['y_true'] for r in records])
            )

    # Select examples
    groups = select_examples_by_group(records, threshold, args.per_group)

    # Determine target conv layer
    target_layer_name = args.layer or find_last_conv_layer_in_base(model)

    # Output dir
    out_dir = os.path.join(final_folder, args.out_dir or 'gradcam_examples')
    save_group_gradcams(
        model,
        groups,
        image_size,
        preprocess_func,
        use_canny,
        target_layer_name,
        out_dir,
        overlay_alpha=args.overlay_alpha,
        blend_mode=args.blend_mode,
        use_mask=(not args.masking_off),
        heatmap_threshold=args.heatmap_threshold,
        heatmap_gamma=args.heatmap_gamma,
        overlay_use_canny=args.overlay_use_canny,
        bg_contrast=args.bg_contrast,
        bg_q_low=args.bg_q_low,
        bg_q_high=args.bg_q_high,
        cam_class=args.cam_class,
        cam_threshold=args.cam_threshold,
    )

    # Save summary
    summary = {
        'model_name': model_name,
        'threshold_used': threshold,
        'target_layer': target_layer_name,
        'counts': {k: len(v) for k, v in groups.items()}
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"Saved Grad-CAM examples to: {out_dir}")
    print(f"Groups sizes: {summary['counts']}")


if __name__ == '__main__':
    main()


