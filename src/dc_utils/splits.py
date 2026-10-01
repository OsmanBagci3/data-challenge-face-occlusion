"""Création reproductible du split train/validation.

Stratégie (cf. EDA étape 1) :
- split aléatoire **stratifié** par la clé `gender × occ_bin × db`,
- seed fixée pour la reproductibilité,
- contrôle automatique : la val doit avoir des effectifs raisonnables dans chaque
  bucket d'occlusion ET dans chaque genre.

NB : on n'utilise **pas** un GroupKFold par identité ici. L'EDA a montré que 66 %
des identités du test sont déjà en train, donc le test évalue sur un mélange
identités vues / non vues. Un split stratifié aléatoire est un meilleur proxy
qu'un GroupSplit strict (qui serait plus pessimiste que le test réel).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def make_strat_key(df: pd.DataFrame) -> pd.Series:
    """Clé de stratification = concat(gender, occ_bin, db) sous forme de string."""
    return (
        df["gender"].astype(str)
        + "|" + df["occ_bin"].astype(str)
        + "|" + df["db"].astype(str)
    )


def stratified_split(
    df: pd.DataFrame,
    val_size: float = 0.1,
    seed: int = 42,
    min_per_class: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Renvoie (df_train, df_val) avec les colonnes du DataFrame d'entrée préservées.

    Les classes de stratification trop rares (< `min_per_class`) sont fusionnées
    pour éviter l'erreur de scikit-learn. Concrètement, ces lignes sont mises
    de force dans le train (elles sont trop rares pour être en val de manière fiable).
    """
    df = df.reset_index(drop=True)
    df["_strat"] = make_strat_key(df)

    counts = df["_strat"].value_counts()
    rare_classes = counts[counts < min_per_class].index
    is_rare = df["_strat"].isin(rare_classes)

    df_safe = df[~is_rare].copy()
    df_forced_train = df[is_rare].copy()

    idx_tr, idx_val = train_test_split(
        df_safe.index,
        test_size=val_size,
        random_state=seed,
        stratify=df_safe["_strat"],
    )
    df_train = pd.concat([df_safe.loc[idx_tr], df_forced_train], ignore_index=True)
    df_val = df_safe.loc[idx_val].reset_index(drop=True)

    df_train = df_train.drop(columns=["_strat"]).reset_index(drop=True)
    df_val = df_val.drop(columns=["_strat"]).reset_index(drop=True)

    return df_train, df_val


def split_summary(df_train: pd.DataFrame, df_val: pd.DataFrame) -> pd.DataFrame:
    """Tableau de contrôle : effectifs et proportions par genre × bucket dans chaque split."""
    rows = []
    for name, sub in [("train", df_train), ("val", df_val)]:
        for g_val, g_lab in [(0.0, "F"), (1.0, "M")]:
            s = sub[sub["gender"] == g_val]
            for b in s["occ_bin"].cat.categories if hasattr(s["occ_bin"], "cat") else sorted(s["occ_bin"].dropna().unique()):
                n = (s["occ_bin"] == b).sum()
                rows.append({"split": name, "gender": g_lab, "bucket": str(b), "n": int(n)})
    return pd.DataFrame(rows).pivot_table(
        index=["gender", "bucket"], columns="split", values="n", fill_value=0
    )
