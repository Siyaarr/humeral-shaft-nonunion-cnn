# Deep learning prediction of humeral shaft nonunion

Code and results for the article *"Deep learning models applied on initial radiographs do not reliably predict union in nonoperative humeral shaft fractures"* (Yücel S, Moilanen M, Kärnä L, Äyrämö S, Reito A; manuscript submitted for publication).

The pipeline compares ten ImageNet-pretrained CNN architectures, each with and without Canny edge detection (20 configurations), using a 10-iteration random search per configuration. The configuration with the highest validation AUC is retrained on the combined training and validation data and evaluated once on the held-out test set. Grad-CAM is used to show which image regions the final model used.

## Data

The radiographs and clinical data are patient data and are **not included**. They cannot be shared publicly under Finnish data protection legislation. Trained model weights are not included either.

`data/test_set_predictions.csv` contains the anonymous test-set predictions of the final model: one row per test radiograph (103 radiographs from 56 patients), with an arbitrary patient number (1–56), the true label (0 = healed, 1 = non-healed) and the predicted probability. These values underlie the test AUC, its confidence interval, Fig 4 and Table 3 of the article.

## Repository contents

| Path | Purpose |
|---|---|
| `src/config.py` | Paths, model list, hyperparameter search space and training settings |
| `src/data_loader.py` | Image loading, letterbox resizing, augmentation, Canny preprocessing, oversampling |
| `src/model.py` | Pretrained backbone with a global-average-pooling, dropout and sigmoid head |
| `src/train.py` | Two-stage training (feature extraction, then fine-tuning) with random search |
| `src/results_analyzer.py` | Summarises the random-search runs |
| `src/final_testing.py` | Selects the winner by validation AUC, retrains it on training + validation data and evaluates it on the test set |
| `src/regenerate_plots.py` | Confusion matrix, ROC and precision–recall curves and probability histogram of the final model (Fig 4) |
| `src/gradcam.py`, `src/generate_gradcam_examples.py` | Grad-CAM heatmaps and example figures (S1 Fig) |
| `src/visualize_data_for_testing.py` | Visual check of preprocessing and augmentation |
| `analysis/auc_ci_bootstrap.py` | 95% CI of the test AUC (patient-level bootstrap, 5,000 resamples) |
| `Lancet_Calculate_95CI.py` | Additional metric and confidence-interval calculations (not used for the values reported in the article) |
| `download_models.py` | Downloads the ImageNet weights in advance, for a machine without internet access |
| `Project.md` | Description of the pipeline |
| `results - FINAL/<model> - Dense_OFF - Canny_<ON/OFF>/run_<n>/` | Hyperparameters and training history of every random-search run |
| `results - FINAL/all_models_all_configurations_test_summary.csv` | Test-set metrics of the best run of each configuration (model selection used validation AUC only) |
| `results - FINAL/final_model/` | Details, training history and test results of the final model; `corrected_plots/` holds Fig 4 and `predictions.npz` |

## Running the code

Python 3.9 and the packages in `requirements.txt` (TensorFlow 2.10.1, Keras 2.10.0).

```
pip install -r requirements.txt
```

The image folders are set in `src/config.py` (`TRAIN_DIR`, `VALIDATION_DIR`, `TEST_DIR`). Each folder contains JPG radiographs named `<patient>_<description>_<label>.jpg`, where the label is `0` (healed) or `1` (non-healed); all radiographs of a patient are in the same folder. Run the commands from the repository folder:

```
python -m src.train --every-model             # random search for all 20 configurations
python -m src.results_analyzer --compare_all  # summary of the runs
python -m src.final_testing                   # winner by validation AUC, retrain, test
python -m src.regenerate_plots                # Fig 4
python -m src.generate_gradcam_examples       # Grad-CAM examples (S1 Fig)
python analysis/auc_ci_bootstrap.py           # AUC 95% CI from data/test_set_predictions.csv
```

`analysis/auc_ci_bootstrap.py` runs without the image data and reproduces the reported AUC of 0.550 (95% CI 0.376 to 0.718).

The classification threshold reported in the article (0.890) was selected as the F1-maximising point of the precision–recall curve on the test set, so the threshold-dependent metrics are best-case estimates (see the article). A radiograph is classified as non-healed when its predicted probability is strictly greater than the threshold, as in `src/final_testing.py`; with the stored threshold in `predictions.npz` this gives the confusion matrix reported in the article (22 true positives, 55 false positives, 5 false negatives, 21 true negatives).

## License

MIT (see `LICENSE`).
