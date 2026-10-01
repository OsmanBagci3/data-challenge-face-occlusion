"""Chargement des CSV + enrichissement avec métadonnées (db, identity, occ_bin).

Les fonctions ici sont *pures* (lecture seule) : elles ne modifient pas les fichiers d'entrée.
Le pré-traitement physique (filtrage des manquants, écriture sur disque) est fait par
le notebook `02_split_and_baseline.ipynb` qui produit `train_clean.csv` etc.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Buckets d'occlusion utilisés pour la stratification du split val.
# Choisis avec l'EDA : ils suivent la queue droite de la distribution.
OCC_BINS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.01]
OCC_LABELS = ["0-5", "5-10", "10-20", "20-30", "30-50", "50-70", "70-100"]


def extract_db(filename: str) -> str:
    """'database3/database3/m.X/...webp' -> 'database3'."""
    return filename.split("/", 1)[0]


def extract_identity(filename: str) -> str:
    """Identité robuste, dépendante de la sous-base.

    - database3 : MID Freebase (vraie identité, ~6 images/personne)
    - database1 : nom de fichier (1 image = 1 sample, pas d'identité partagée)
    - database2 : sous-dossier + nom (volume négligeable)
    """
    parts = filename.split("/")
    if not parts:
        return "unk"
    if parts[0] == "database3" and len(parts) >= 3:
        return f"db3:{parts[2]}"
    if parts[0] == "database1" and len(parts) >= 2:
        return f"db1:{parts[1]}"
    if parts[0] == "database2" and len(parts) >= 4:
        return f"db2:{parts[2]}/{parts[3]}"
    return f"unk:{filename}"


def add_metadata(df: pd.DataFrame, has_target: bool = True) -> pd.DataFrame:
    """Ajoute les colonnes db, identity, et (si applicable) occ_bin."""
    df = df.copy()
    df["db"] = df["filename"].map(extract_db)
    df["identity"] = df["filename"].map(extract_identity)
    if has_target and "FaceOcclusion" in df.columns:
        df["occ_bin"] = pd.cut(
            df["FaceOcclusion"], bins=OCC_BINS, labels=OCC_LABELS, include_lowest=True
        )
    return df


def load_train(csv_path: Path) -> pd.DataFrame:
    """Charge le CSV train brut et ajoute les métadonnées."""
    df = pd.read_csv(csv_path)
    return add_metadata(df, has_target=True)


def load_test(csv_path: Path) -> pd.DataFrame:
    """Charge le CSV test brut et ajoute les métadonnées (sans cible)."""
    df = pd.read_csv(csv_path)
    return add_metadata(df, has_target=False)
