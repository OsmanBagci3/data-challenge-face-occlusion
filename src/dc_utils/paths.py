"""Chemins absolus du projet, dérivés de la racine du repo."""
from pathlib import Path

# Racine du projet : .../data_challenge/
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR  = ROOT / "data"
IMAGE_DIR = ROOT / "crops" / "Crop_224_5fp_100K"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV  = DATA_DIR / "test_students.csv"

# Fichiers générés par le pipeline (étape 2)
TRAIN_CLEAN_CSV = DATA_DIR / "train_clean.csv"
TEST_CLEAN_CSV  = DATA_DIR / "test_clean.csv"
TRAIN_SPLIT_CSV = DATA_DIR / "train_split.csv"
VAL_SPLIT_CSV   = DATA_DIR / "val_split.csv"

ARTIFACTS_DIR = ROOT / "artifacts"
SUBMISSIONS_DIR = ROOT / "submissions"
