"""Classical baselines on per-cell phase patches.

Regressing all 800 outputs from the full 224x224 = 50,176-pixel image with only
60 training images is hopelessly under-determined for a linear model, so the
linear baselines are applied cell-wise: each unit cell's 12x12 phase patch is a
feature vector (60 images x 400 cells = 24,000 training rows).
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .data import DELTA_MAX, IMG, N_CELLS
from .metrics import summarize

PATCH = 12


def cell_patches(X: np.ndarray) -> np.ndarray:
    """(N, 224, 224) -> (N, 20, 20, 144) raw-phase patches, one per unit cell."""
    starts = [min(int(np.floor(i * IMG / N_CELLS)), IMG - PATCH) for i in range(N_CELLS)]
    out = np.empty((len(X), N_CELLS, N_CELLS, PATCH * PATCH), dtype=np.float32)
    for i, si in enumerate(starts):
        for j, sj in enumerate(starts):
            out[:, i, j] = X[:, si:si + PATCH, sj:sj + PATCH].reshape(len(X), -1)
    return out


def _rows(X, Y):
    F = cell_patches(X).reshape(-1, PATCH * PATCH)
    T = Y.transpose(0, 2, 3, 1).reshape(-1, 2)  # (N*400, 2) classes for dx, dy
    return F, T


def _to_maps(pred_rows: np.ndarray, n: int) -> np.ndarray:
    return pred_rows.reshape(n, N_CELLS, N_CELLS, 2).transpose(0, 3, 1, 2)


def majority(data, tr, va, te) -> dict:
    _, T = _rows(data["X"][tr], data["Y"][tr])
    mode = [np.bincount(T[:, k]).argmax() for k in range(2)]
    pred = np.empty_like(data["Y"][te])
    pred[:, 0], pred[:, 1] = mode
    return summarize(pred, data["Y"][te])


def linear_regression(data, tr, va, te) -> dict:
    F, T = _rows(data["X"][tr], data["Y"][tr])
    reg = LinearRegression().fit(F, (T - DELTA_MAX).astype(np.float64))
    Fte, _ = _rows(data["X"][te], data["Y"][te])
    p = np.clip(np.round(reg.predict(Fte)), -DELTA_MAX, DELTA_MAX).astype(np.int64) + DELTA_MAX
    return summarize(_to_maps(p, len(te)), data["Y"][te])


def linear_classification(data, tr, va, te, Cs=(0.01, 0.1, 1.0)) -> dict:
    """LIBLINEAR linear SVM (one-vs-rest, 31 classes) per target; C chosen on validation."""
    F, T = _rows(data["X"][tr], data["Y"][tr])
    Fva, Tva = _rows(data["X"][va], data["Y"][va])
    Fte, _ = _rows(data["X"][te], data["Y"][te])
    sc = StandardScaler().fit(F)
    F, Fva, Fte = sc.transform(F), sc.transform(Fva), sc.transform(Fte)
    preds, chosen = [], []
    for k in range(2):
        best = (-1.0, None, None)
        for C in Cs:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf = LinearSVC(C=C, dual=False, max_iter=2000).fit(F, T[:, k])
            acc = float((clf.predict(Fva) == Tva[:, k]).mean())
            if acc > best[0]:
                best = (acc, C, clf)
        chosen.append(best[1])
        preds.append(best[2].predict(Fte))
    res = summarize(_to_maps(np.stack(preds, 1), len(te)), data["Y"][te])
    res["C"] = chosen
    return res
