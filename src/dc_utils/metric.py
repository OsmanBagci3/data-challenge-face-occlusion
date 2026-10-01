"""Implémentation officielle de la métrique du Data Challenge.

Formule :
    Err   = sum_i w_i (p_i - GT_i)^2  /  sum_i w_i
    w_i   = 1/30 + GT_i
    Score = (Err_F + Err_M) / 2  +  |Err_F - Err_M|

Conventions :
- gender == 1.0  -> homme
- gender == 0.0  -> femme
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def weighted_error(pred: np.ndarray, target: np.ndarray) -> float:
    """MSE pondérée par w_i = 1/30 + GT_i (formule officielle du challenge)."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    w = 1.0 / 30.0 + target
    num = np.sum(w * (pred - target) ** 2)
    den = np.sum(w)
    return float(num / den)


def compute_score(
    df: pd.DataFrame,
    pred_col: str = "pred",
    target_col: str = "target",
    gender_col: str = "gender",
) -> dict:
    """Calcule (score, err_F, err_M, gap) à partir d'un DataFrame de prédictions.

    Le DataFrame doit contenir au moins les colonnes pred_col, target_col, gender_col.
    Retourne un dict pour pouvoir être logué facilement.
    """
    sub_F = df[df[gender_col] == 0.0]
    sub_M = df[df[gender_col] == 1.0]
    err_F = weighted_error(sub_F[pred_col].values, sub_F[target_col].values)
    err_M = weighted_error(sub_M[pred_col].values, sub_M[target_col].values)
    gap = abs(err_F - err_M)
    score = 0.5 * (err_F + err_M) + gap
    return {
        "score": score,
        "err_F": err_F,
        "err_M": err_M,
        "gap": gap,
        "n_F": len(sub_F),
        "n_M": len(sub_M),
    }


# ---- Versions PyTorch pour utilisation dans la loss / pendant l'entraînement ----

def torch_weighted_mse(pred, target, eps: float = 1e-12):
    """MSE pondérée — utilisable comme loss alignée sur la métrique du challenge.

    Args:
        pred, target : tensors 1D de même taille (occlusion ∈ [0, 1]).
    Returns:
        scalar tensor.
    """
    import torch
    w = 1.0 / 30.0 + target
    num = (w * (pred - target) ** 2).sum()
    den = w.sum().clamp_min(eps)
    return num / den
