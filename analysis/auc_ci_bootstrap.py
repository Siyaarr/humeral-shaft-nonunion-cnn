"""95% confidence interval of the test AUC ROC (final VGG16 model, 103 test radiographs from 56 patients).

Reads data/test_set_predictions.csv (one row per test radiograph: anonymous patient number,
true label and predicted probability) and prints the AUC with three 95% CIs:
DeLong (per radiograph), bootstrap per radiograph, and bootstrap per patient
(5,000 resamples, seed 2026). The article reports the patient-level bootstrap CI.

Usage (from the repository folder):
    python analysis/auc_ci_bootstrap.py [path/to/test_set_predictions.csv]
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "..", "data", "test_set_predictions.csv")


def roc_auc_score(y, p):
    pos, neg = p[y == 1], p[y == 0]
    return ((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])).mean()


def delong(y, p):
    pos, neg = p[y == 1], p[y == 0]
    m, n = len(pos), len(neg)
    psi = (pos[:, None] > neg[None, :]).astype(float) + 0.5 * (pos[:, None] == neg[None, :])
    auc = psi.mean()
    se = np.sqrt(psi.mean(1).var(ddof=1) / m + psi.mean(0).var(ddof=1) / n)
    return auc, auc - 1.96 * se, auc + 1.96 * se


def main(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    pid = np.array([int(r["patient"]) for r in rows])
    y = np.array([int(r["true_label"]) for r in rows])
    p = np.array([float(r["predicted_probability"]) for r in rows])
    print("AUC %.4f  (%d radiographs, %d non-healed, %d patients)" % (roc_auc_score(y, p), len(y), y.sum(), len(set(pid))))
    print("DeLong per-radiograph 95%% CI: %.3f (%.3f to %.3f)" % delong(y, p))

    rng = np.random.default_rng(2026)
    b = []
    for _ in range(5000):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            b.append(roc_auc_score(y[i], p[i]))
    print("Bootstrap per radiograph 95%% CI: %.3f to %.3f" % tuple(np.percentile(b, [2.5, 97.5])))

    patients = np.unique(pid)
    index = {u: np.where(pid == u)[0] for u in patients}
    b = []
    for _ in range(5000):
        chosen = rng.choice(patients, len(patients))
        i = np.concatenate([index[u] for u in chosen])
        if y[i].min() != y[i].max():
            b.append(roc_auc_score(y[i], p[i]))
    print("Bootstrap per patient 95%% CI: %.3f to %.3f" % tuple(np.percentile(b, [2.5, 97.5])))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
