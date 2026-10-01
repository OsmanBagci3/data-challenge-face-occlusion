"""Build notebook 05_tta_ensemble_calibration.ipynb."""
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / '05_tta_ensemble_calibration.ipynb'


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


cells = []

cells.append(md(r"""# Étape 5 — TTA + Ensemble + Calibration (Phase A)

**Constat étape 4** :

| Modèle | val score | err_F | err_M | gap | gap / score |
|---|---|---|---|---|---|
| baseline mbv3s | 0.00122 | 0.00064 | 0.00103 | 0.00039 | 32 % |
| F1 effb0 | 0.00120 | 0.00072 | 0.00104 | 0.00032 | 27 % |
| **F2 convnext_tiny** | **0.00116** | 0.00043 | 0.00092 | 0.00049 | **42 %** |

**Cible leaderboard mise à jour : 0.00092**.

**Levier principal : le gap genre.** Le score est $\frac{Err_F + Err_M}{2} + |Err_F - Err_M|$, donc le gap est **doublement compté**. Sur convnext il représente 42 % du score → si on ramène err_F vers err_M (ou vice-versa), gain massif sans toucher au modèle.

**Plan Phase A — gains gratuits (aucun ré-entraînement)** :
1. **TTA flip horizontal** à l'inférence (moyenne pred(x) et pred(flip(x))) — gain typique 2-5 %.
2. **Ensemble F1+F2** (moyenne des sigmoid) — diversité utile car archi différentes.
3. **Calibration isotonique par genre** fittée sur val — attaque le gap directement.

**Méthodo honnête pour la calibration** : on utilise une **5-fold CV sur val** pour estimer le score post-calibration sans biais (sinon on évalue sur les données utilisées pour fitter → optimiste). Pour le pipeline final d'inférence, on refittera l'isotonique sur 100 % de val.

⚠️ **Rappel shift train/test** (chiffre du collègue : ResNet50 0.00133 local → 0.0012 soumission). Ne pas sur-optimiser le score val.
"""))

cells.append(md("## 1. Setup"))

cells.append(code("""import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader

from dc_utils.paths import IMAGE_DIR, ARTIFACTS_DIR
from dc_utils.training import OcclusionDataset, default_eval_tf
from dc_utils.metric import compute_score, weighted_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device =', device, '| timm', timm.__version__, '| torch', torch.__version__)

EXP_DIR = ROOT / 'experiments'
df_val = pd.read_csv(ROOT / 'data' / 'val_split.csv')
print('val:', len(df_val), '| n_F:', (df_val.gender == 0.0).sum(), '| n_M:', (df_val.gender == 1.0).sum())
"""))

cells.append(md("## 2. Reload F1 (efficientnet_b0) et F2 (convnext_tiny)"))

cells.append(code("""# Même TimmRegressor que l'étape 4. On charge des poids déjà entraînés,
# donc bias_init est inutile ici (mais on garde l'API cohérente).
class TimmRegressor(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool = False, bias_init: float | None = None):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=1)
        if bias_init is not None:
            head = self.backbone.get_classifier()
            with torch.no_grad():
                head.bias.fill_(bias_init); head.weight.mul_(0.01)
    def forward(self, x):
        return torch.sigmoid(self.backbone(x)).squeeze(-1)

def load_ckpt(backbone_name: str, ckpt_path: Path) -> nn.Module:
    m = TimmRegressor(backbone_name, pretrained=False).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    m.load_state_dict(state['model'])
    m.eval()
    print(f'loaded {ckpt_path.name} | backbone={backbone_name} | best ep={state.get("epoch")} | val={state.get("val")}')
    return m

model_F1 = load_ckpt('efficientnet_b0', ARTIFACTS_DIR / 'exp_F1_effb0.pt')
model_F2 = load_ckpt('convnext_tiny',   ARTIFACTS_DIR / 'exp_F2_convnext.pt')
"""))

cells.append(md(r"""## 3. Helper de prédiction (avec ou sans TTA flip horizontal)

Fonction qui, étant donné un modèle et un DataLoader, retourne un `np.ndarray` de prédictions dans le **même ordre** que `df_val`. Pour le TTA flip on calcule $\frac{1}{2}(\text{pred}(x) + \text{pred}(\text{flip}(x)))$.

`OcclusionDataset(has_target=True)` retourne `(x, y, gender, filename)`. On utilise `batch[0]` (input) et `batch[-1]` (filenames)."""))

cells.append(code("""@torch.inference_mode()
def predict(model, dl, device, tta_flip: bool = False, use_amp: bool = True):
    model.eval()
    preds, fns = [], []
    ctx = torch.amp.autocast('cuda', enabled=use_amp)
    for batch in dl:
        X = batch[0].to(device, non_blocking=True)
        fn = batch[-1]
        with ctx:
            p = model(X).float()
            if tta_flip:
                p_flip = model(torch.flip(X, dims=[-1])).float()
                p = 0.5 * (p + p_flip)
        preds.append(p.cpu().numpy()); fns.extend(fn)
    return np.concatenate(preds), fns

ds_val = OcclusionDataset(df_val, IMAGE_DIR, default_eval_tf(), has_target=True)
dl_val = DataLoader(ds_val, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

def df_with_pred(df_base: pd.DataFrame, preds: np.ndarray, fns: list[str]) -> pd.DataFrame:
    out = pd.DataFrame({'filename': fns, 'pred': preds})
    out = (df_base[['filename','FaceOcclusion','gender']]
           .rename(columns={'FaceOcclusion':'target'})
           .merge(out, on='filename'))
    assert len(out) == len(df_base), f'merge size mismatch: {len(out)} vs {len(df_base)}'
    return out
"""))

cells.append(md("## 4. Recettes de base — F1, F2, avec/sans TTA"))

cells.append(code("""import time

RECIPES: dict[str, pd.DataFrame] = {}

def run(name: str, model, tta_flip: bool):
    t0 = time.time()
    p, fns = predict(model, dl_val, device, tta_flip=tta_flip)
    df_pred = df_with_pred(df_val, p, fns)
    s = compute_score(df_pred)
    dt = time.time() - t0
    print(f'{name:25s} | score {s["score"]:.5f} | err_F {s["err_F"]:.5f} | err_M {s["err_M"]:.5f} | gap {s["gap"]:.5f} | {dt:.0f}s')
    RECIPES[name] = df_pred
    return s

_ = run('F1_effb0',          model_F1, tta_flip=False)
_ = run('F2_convnext',       model_F2, tta_flip=False)
_ = run('F1_effb0_TTA',      model_F1, tta_flip=True)
_ = run('F2_convnext_TTA',   model_F2, tta_flip=True)
"""))

cells.append(md("**Sanity** : F1/F2 sans TTA doivent reproduire les scores du training (~0.00120 / ~0.00116, à ±1e-5)."))

cells.append(md("## 5. Ensemble F1 + F2 (moyenne pondérée)"))

cells.append(code("""def ensemble_preds(*dfs: pd.DataFrame, weights: list[float] | None = None) -> pd.DataFrame:
    base = dfs[0][['filename','target','gender']].copy()
    P = np.stack([d.set_index('filename').loc[base['filename'], 'pred'].values for d in dfs])
    if weights is None:
        weights = [1.0/len(dfs)] * len(dfs)
    base['pred'] = (np.array(weights)[:,None] * P).sum(axis=0)
    return base

def show(name, df_pred):
    s = compute_score(df_pred)
    print(f'{name:32s} | score {s["score"]:.5f} | err_F {s["err_F"]:.5f} | err_M {s["err_M"]:.5f} | gap {s["gap"]:.5f}')
    RECIPES[name] = df_pred
    return s

_ = show('ENS_F1+F2',           ensemble_preds(RECIPES['F1_effb0'],     RECIPES['F2_convnext']))
_ = show('ENS_F1+F2_TTA',       ensemble_preds(RECIPES['F1_effb0_TTA'], RECIPES['F2_convnext_TTA']))
# Variantes pondérées : convnext est meilleur en val seul, on lui donne plus de poids.
_ = show('ENS_0.3F1+0.7F2_TTA', ensemble_preds(RECIPES['F1_effb0_TTA'], RECIPES['F2_convnext_TTA'], weights=[0.3, 0.7]))
_ = show('ENS_0.4F1+0.6F2_TTA', ensemble_preds(RECIPES['F1_effb0_TTA'], RECIPES['F2_convnext_TTA'], weights=[0.4, 0.6]))
"""))

cells.append(md(r"""## 6. Calibration isotonique par genre — 5-fold CV honnête

**Idée** : pour chaque genre $g \in \{F, M\}$, on apprend une fonction monotone $f_g : [0,1] \to [0,1]$ qui mappe la prédiction brute à la prédiction calibrée. Non-paramétrique mais avec une seule contrainte (monotonie), donc peu de risque d'overfit.

**Pourquoi par genre** : le gap entre err_F et err_M peut venir d'un biais systématique (ex : sur-prédiction sur F dans certains buckets). Une isotonique par genre corrige ça **directement**.

**Méthodo honnête** : 5-fold CV sur val :
- Fold k : isotonique fittée sur 4/5, appliquée sur le 1/5 hold-out.
- Score = `compute_score` sur les 5 hold-outs concaténés → estimation non-biaisée.

⚠️ Si l'isotonique fait un trop gros gain en CV honnête, c'est suspect (sur-ajustement spécifique à la val). On compare aussi à l'iso globale (sans split par genre)."""))

cells.append(code("""from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

def kfold_isotonic_per_gender(df_pred: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    df = df_pred.reset_index(drop=True).copy()
    df['pred_cal'] = np.nan
    for g_value in [0.0, 1.0]:
        idx_g = df.index[df['gender'] == g_value].to_numpy()
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr_local, te_local in kf.split(idx_g):
            tr = idx_g[tr_local]; te = idx_g[te_local]
            iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
            iso.fit(df.loc[tr,'pred'].values, df.loc[tr,'target'].values)
            df.loc[te, 'pred_cal'] = iso.predict(df.loc[te, 'pred'].values)
    out = df[['filename','target','gender']].copy()
    out['pred'] = df['pred_cal'].values
    return out

def calib(name_in: str, name_out: str):
    df_cal = kfold_isotonic_per_gender(RECIPES[name_in])
    return show(name_out, df_cal)

print('--- Calibration isotonique par genre (5-fold honnête) ---')
_ = calib('F2_convnext_TTA',      'F2_TTA + iso(g)')
_ = calib('F1_effb0_TTA',         'F1_TTA + iso(g)')
_ = calib('ENS_F1+F2_TTA',        'ENS_TTA + iso(g)')
_ = calib('ENS_0.3F1+0.7F2_TTA',  'ENS_0.3/0.7_TTA + iso(g)')
_ = calib('ENS_0.4F1+0.6F2_TTA',  'ENS_0.4/0.6_TTA + iso(g)')
"""))

cells.append(md("## 7. Sanity — calibration isotonique **globale** (sans split par genre)"))

cells.append(code("""def kfold_isotonic_global(df_pred: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    df = df_pred.reset_index(drop=True).copy()
    df['pred_cal'] = np.nan
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in kf.split(df):
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        iso.fit(df.loc[tr,'pred'].values, df.loc[tr,'target'].values)
        df.loc[te, 'pred_cal'] = iso.predict(df.loc[te, 'pred'].values)
    out = df[['filename','target','gender']].copy()
    out['pred'] = df['pred_cal'].values
    return out

print('--- Calibration isotonique globale (5-fold) ---')
for name_in in ['F2_convnext_TTA', 'ENS_F1+F2_TTA', 'ENS_0.3F1+0.7F2_TTA']:
    df_cal = kfold_isotonic_global(RECIPES[name_in])
    show(f'{name_in} + iso(global)', df_cal)
"""))

cells.append(md("## 8. Tableau de synthèse — toutes les recettes"))

cells.append(code("""rows = []
for name, df_pred in RECIPES.items():
    s = compute_score(df_pred)
    rows.append({'recipe': name, **s})
df_recipes = pd.DataFrame(rows).sort_values('score').reset_index(drop=True)
ref_score = df_recipes.loc[df_recipes['recipe']=='F2_convnext','score'].iloc[0]
df_recipes['gain_vs_F2'] = df_recipes['score'] - ref_score
df_recipes['gain_%']     = (-df_recipes['gain_vs_F2'] / ref_score * 100).round(2)
df_recipes['gap_%']      = (df_recipes['gap'] / df_recipes['score'] * 100).round(1)
df_recipes.to_csv(EXP_DIR / 'step5_phaseA_recipes.csv', index=False)
df_recipes
"""))

cells.append(code("""import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, max(3.5, 0.35*len(df_recipes))))
ax.barh(df_recipes['recipe'], df_recipes['score'], color='steelblue')
ax.axvline(0.00092, color='red', ls='--', label='leaderboard 0.00092')
ax.axvline(0.00116, color='gray', ls=':', label='F2 brut')
ax.set_xlabel('val score (plus bas = mieux)')
ax.set_title('Phase A — TTA, ensemble, calibration')
ax.legend(loc='lower right'); ax.invert_yaxis()
plt.tight_layout(); plt.show()
"""))

cells.append(md("""## 9. Sauvegarde du pipeline d'inférence final

On choisit la **meilleure recette** du tableau, on **refit** l'isotonique sur 100 % de val (plus de hold-out, c'est le pipeline final), et on sauvegarde tous les artefacts pour la soumission test."""))

cells.append(code("""import pickle, re

# Choix automatique de la meilleure recette (modifiable manuellement)
BEST_RECIPE = df_recipes.iloc[0]['recipe']
print(f'Best recipe (val) : {BEST_RECIPE}  -> score {df_recipes.iloc[0]["score"]:.5f}')

USE_CALIB = ' + iso' in BEST_RECIPE
USE_TTA   = '_TTA' in BEST_RECIPE
USE_ENS   = BEST_RECIPE.startswith('ENS')
print(f'USE_TTA={USE_TTA} | USE_ENS={USE_ENS} | USE_CALIB={USE_CALIB}')

def fit_iso_per_gender_full(df_pred: pd.DataFrame) -> dict:
    iso_by_g = {}
    for g_value in [0.0, 1.0]:
        m = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        sub = df_pred[df_pred['gender'] == g_value]
        m.fit(sub['pred'].values, sub['target'].values)
        iso_by_g[g_value] = m
    return iso_by_g

def fit_iso_global_full(df_pred: pd.DataFrame) -> IsotonicRegression:
    m = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    m.fit(df_pred['pred'].values, df_pred['target'].values)
    return m

def apply_iso_per_gender(df_pred: pd.DataFrame, iso_by_g: dict) -> pd.DataFrame:
    out = df_pred.copy()
    for g_value, m in iso_by_g.items():
        mask = out['gender'] == g_value
        out.loc[mask, 'pred'] = m.predict(out.loc[mask, 'pred'].values)
    return out

# Recette de base (sans le suffixe ' + iso(*)')
base_recipe = re.sub(r' \\+ iso.*$', '', BEST_RECIPE)
df_uncal = RECIPES[base_recipe]

iso_per_gender = None
iso_global     = None
if USE_CALIB:
    if 'iso(global)' in BEST_RECIPE:
        iso_global = fit_iso_global_full(df_uncal)
        df_final_val = df_uncal.copy()
        df_final_val['pred'] = iso_global.predict(df_final_val['pred'].values)
    else:
        iso_per_gender = fit_iso_per_gender_full(df_uncal)
        df_final_val = apply_iso_per_gender(df_uncal, iso_per_gender)
else:
    df_final_val = df_uncal

print('Pipeline final (refit iso sur 100% val) :', compute_score(df_final_val))

ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)
pipeline_path = ARTIFACTS_DIR / 'pipeline_phaseA.pkl'
with open(pipeline_path, 'wb') as f:
    pickle.dump({
        'best_recipe': BEST_RECIPE,
        'use_tta': USE_TTA,
        'use_ensemble': USE_ENS,
        'use_calib': USE_CALIB,
        'iso_per_gender': iso_per_gender,
        'iso_global': iso_global,
        'val_score_final': compute_score(df_final_val),
        'ckpts': {
            'F1_effb0':    str(ARTIFACTS_DIR / 'exp_F1_effb0.pt'),
            'F2_convnext': str(ARTIFACTS_DIR / 'exp_F2_convnext.pt'),
        },
    }, f)
print(f'pipeline saved -> {pipeline_path}')
df_final_val.to_csv(EXP_DIR / 'preds_val_pipelineA.csv', index=False)
"""))

cells.append(md(r"""## 10. Conclusions & next steps

*(à remplir après exécution)*

- Meilleure recette val : ?
- Gain TTA seul : ? %
- Gain ensemble F1+F2 seul : ? %
- Gain calibration iso par genre : ? %
- Réduction du gap : avant ? → après ?
- Score val final pipeline : ?

**Phase B (planifiée)** — diversité d'ensemble, ré-entraînement :
1. convnext_tiny seed=1 (réduire variance) → ensemble à 3 modèles.
2. Si temps : backbone supplémentaire (efficientnet_b3 ou convnext_small) pour diversité.
3. Ré-entraînement final sur train+val (plus de hold-out, archi figée).

**Phase C** — première soumission :
1. Pipeline test : prédire sur test_clean.csv avec la même recette (TTA + ensemble + iso).
2. Soumettre 1 fois pour calibrer notre local→leaderboard.
3. Selon écart, ajuster (plus de régularisation si overfit, plus de capacité si underfit).
"""))

with open(NB_PATH, 'w') as f:
    json.dump({
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python (.venv)", "language": "python", "name": "venv"},
            "language_info": {"name": "python", "version": "3.12.3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }, f, indent=1, ensure_ascii=False)

print(f'OK {NB_PATH} -> {len(cells)} cells')
