"""Accuracy metrics on per-cell (dx, dy) class predictions."""

from __future__ import annotations

import numpy as np


def summarize(pred: np.ndarray, true: np.ndarray) -> dict:
    """pred/true: integer class maps of shape (N, 2, 20, 20).

    acc     -- exact class match, averaged over all cells and over dx and dy
               (the paper's 'Accuracy');
    acc_x / acc_y -- per-parameter exact accuracy;
    acc_pm1 -- within +-1 nm;
    mae_nm  -- mean absolute error in nm.
    """
    err = np.abs(pred.astype(np.int64) - true.astype(np.int64))
    return dict(
        acc=float((err == 0).mean()),
        acc_x=float((err[:, 0] == 0).mean()),
        acc_y=float((err[:, 1] == 0).mean()),
        acc_pm1=float((err <= 1).mean()),
        mae_nm=float(err.mean()),
    )
