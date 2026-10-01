"""Build notebook 06_phaseB_more_models.ipynb."""
import json, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / '06_phaseB_more_models.ipynb'


def md(src):
    return {"cell_type":"markdown","id":uuid.uuid4().hex[:8],"metadata":{},"source":src.splitlines(keepends=True)}
def code(src):
    return {"cell_type":"code","id":uuid.uuid4().hex[:8],"metadata":{},"execution_count":None,"outputs":[],"source":src.splitlines(keepends=True)}


cells = []

cells.append(md(r"""# Étape 6 — Phase B : diversité d'ensemble

**Constat Phase A** : meilleure recette val = `ENS_0.4F1+0.6F2_TTA` à **0.001029** (-11.6 % vs F2 seul). Cible leaderboard 0.00092 → reste -11 % à grappiller.

**Décisions issues de Phase A** :
- Calibration isotonique (par genre OU globale) **dégrade** systématiquement en CV honnête. La loss `torch_weighted_mse` aligne déjà les prédictions sur la métrique → pas de mismatch à corriger. **On abandonne la calibration.**
- L'ensemble est le levier qui marche → on ajoute des modèles diversifiés.
- Le gap genre (~38 % du score) est *structurel* (les hommes ont des poids $w_i = 1/30 + GT_i$ plus grands car cibles plus élevées en moyenne), pas un biais de calibration. La seule façon de le réduire est de **mieux prédire les forts taux d'occlusion**.

**Plan Phase B** — entraîner 2 nouveaux modèles complémentaires :

| Modèle | Pourquoi | Temps 3090 estimé |
|---|---|---|
| **M3 = convnext_tiny seed=1** | même archi, autre seed → réduit la variance d'ensemble | ~22 min |
| **M4 = efficientnet_b3** | famille différente, +capacité (12M params vs 4M pour B0) | ~30 min |

**Total** : ~50 min. À la fin, ensemble à 4 modèles + recherche des poids optimaux par scipy.

**Hyperparams** : on garde la recette qui marche pour F1/F2 :
- `bias_init = logit(p0)` (indispensable pour ne pas saturer la sigmoid)
- `lr=1e-4`, `weight_decay=1e-4`, 8 epochs, cosine, AMP
- `batch_size=128` pour effb3 (12M params, OK sur 3090), `batch_size=64` pour convnext_tiny

⚠️ **Reminder** : tester en val ne suffit pas (shift train/test). On vise un **ensemble robuste**, pas un overfit val.
"""))

cells.append(md("## 1. Setup — mêmes bases que step 4"))

cells.append(code("""import sys, time
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
from dc_utils.training import TrainConfig, train_model, OcclusionDataset, default_eval_tf
from dc_utils.metric import compute_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device =', device, '| timm', timm.__version__, '| torch', torch.__version__)

EXP_DIR = ROOT / 'experiments'
df_train = pd.read_csv(ROOT / 'data' / 'train_split.csv')
df_val   = pd.read_csv(ROOT / 'data' / 'val_split.csv')
print('train:', len(df_train), '| val:', len(df_val))

# Mêmes définitions qu'étape 4
class TimmRegressor(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool = True, bias_init: float | None = None):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=1)
        if bias_init is not None:
            head = self.backbone.get_classifier()
            with torch.no_grad():
                head.bias.fill_(bias_init); head.weight.mul_(0.01)
    def forward(self, x):
        return torch.sigmoid(self.backbone(x)).squeeze(-1)

_p0 = float(df_train['FaceOcclusion'].mean())
HEAD_BIAS_INIT = float(np.log(_p0 / (1.0 - _p0)))
print(f'mean target = {_p0:.4f}  ->  bias_init = logit(p0) = {HEAD_BIAS_INIT:.4f}')

def fresh_model(backbone_name: str, seed: int = 42, bias_init: float | None = HEAD_BIAS_INIT):
    torch.manual_seed(seed); np.random.seed(seed)
    m = TimmRegressor(backbone_name, bias_init=bias_init).to(device)
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f'{backbone_name} (seed={seed}): {n/1e6:.2f}M params | bias_init={bias_init:.3f}')
    return m

RESULTS: list[dict] = []
def log_run(name, history, best, notes=''):
    history.to_csv(EXP_DIR / f'history_{name}.csv', index=False)
    RESULTS.append({'name': name, 'best_epoch': best['epoch'], 'val_score': best['score'],
                    'err_F': best['err_F'], 'err_M': best['err_M'], 'gap': best['gap'], 'notes': notes})
    print(f'\\n>>> {name}: val_score = {best["score"]:.5f} (epoch {best["epoch"]})\\n')
"""))

cells.append(md(r"""## 2. M3 — convnext_tiny seed=1

**Pourquoi** : même archi que F2 mais autre initialisation aléatoire (seed différent). Donne un modèle dont les *erreurs* sur val sont **partiellement décorrélées** de celles de F2, donc l'ensemble {F2, M3} sera plus robuste qu'un seul modèle moyenné. C'est l'astuce classique de Snapshot/Multi-seed averaging.

**Coût** : ~22 min sur 3090. Score val attendu : ~0.00116 ± 0.00005 (proche de F2). Ce qui compte ce n'est PAS d'avoir un meilleur score solo, c'est la décorrélation."""))

cells.append(code("""model = fresh_model('convnext_tiny', seed=1)
cfg = TrainConfig(epochs=8, lr=1e-4, weight_decay=1e-4, batch_size=64, use_amp=True)
hist, best = train_model(
    model, df_train, df_val, IMAGE_DIR, cfg, device,
    save_path=ARTIFACTS_DIR / 'exp_M3_convnext_seed1.pt', log_prefix='[M3] ',
)
log_run('M3_convnext_seed1', hist, best, notes='convnext_tiny, seed=1, 8ep, BS=64, 3090/AMP')
"""))

cells.append(md(r"""## 3. M4 — efficientnet_b3

**Pourquoi** : famille EfficientNet (compound scaling) avec ~3× plus de params que B0 (12M vs 4M) → plus de capacité pour mieux modéliser les forts taux d'occlusion. Archi *différente* de convnext (moderne CNN vs scaled MBConv) → bonne diversité d'ensemble.

**Note** : EfficientNet-B3 est conçu pour 300×300 nativement, mais nos images sont en 224. timm gère ça sans souci (global pooling à la fin). On reste à 224 pour cohérence pipeline.

**Coût** : ~30 min sur 3090. Score val attendu : ~0.00115-0.00120 (légèrement meilleur que B0)."""))

cells.append(code("""model = fresh_model('efficientnet_b3', seed=42)
# 12M params -> on garde batch_size=128 (le head GAP a peu d'impact mémoire vs B0)
cfg = TrainConfig(epochs=8, lr=1e-4, weight_decay=1e-4, batch_size=128, use_amp=True)
hist, best = train_model(
    model, df_train, df_val, IMAGE_DIR, cfg, device,
    save_path=ARTIFACTS_DIR / 'exp_M4_effb3.pt', log_prefix='[M4] ',
)
log_run('M4_effb3', hist, best, notes='efficientnet_b3, seed=42, 8ep, BS=128, 3090/AMP')
"""))

cells.append(md("## 4. Reload des 4 modèles — F1, F2, M3, M4"))

cells.append(code("""def load_ckpt(backbone_name: str, ckpt_path: Path) -> nn.Module:
    m = TimmRegressor(backbone_name, pretrained=False).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    m.load_state_dict(state['model'])
    m.eval()
    print(f'loaded {ckpt_path.name} | best ep={state.get("epoch")} | val={state.get("val")}')
    return m

models = {
    'F1_effb0':           load_ckpt('efficientnet_b0', ARTIFACTS_DIR / 'exp_F1_effb0.pt'),
    'F2_convnext':        load_ckpt('convnext_tiny',   ARTIFACTS_DIR / 'exp_F2_convnext.pt'),
    'M3_convnext_seed1':  load_ckpt('convnext_tiny',   ARTIFACTS_DIR / 'exp_M3_convnext_seed1.pt'),
    'M4_effb3':           load_ckpt('efficientnet_b3', ARTIFACTS_DIR / 'exp_M4_effb3.pt'),
}
"""))

cells.append(md("## 5. Prédictions val de chaque modèle (avec TTA flip horizontal)"))

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

# Récupère la liste des filenames une fois (ordre val, shuffle=False)
_, FILENAMES = predict(next(iter(models.values())), dl_val, device, tta_flip=False)

PREDS = {}     # name -> np.ndarray (N,) pred TTA
for name, m in models.items():
    t0 = time.time()
    p, fns = predict(m, dl_val, device, tta_flip=True)
    assert fns == FILENAMES, 'order mismatch'
    PREDS[name] = p
    # score solo (TTA)
    df_p = pd.DataFrame({'filename': FILENAMES, 'pred': p}).merge(
        df_val[['filename','FaceOcclusion','gender']].rename(columns={'FaceOcclusion':'target'}), on='filename')
    s = compute_score(df_p)
    print(f'{name:22s} TTA | score {s["score"]:.5f} | err_F {s["err_F"]:.5f} | err_M {s["err_M"]:.5f} | gap {s["gap"]:.5f} | {time.time()-t0:.0f}s')
"""))

cells.append(md(r"""## 6. Diversité — corrélation des erreurs entre modèles

Avant de chercher les poids optimaux, on regarde si nos 4 modèles sont vraiment diversifiés. **Si deux modèles ont leurs erreurs corrélées à 0.99, l'un des deux est redondant**.

On calcule la corrélation de Pearson sur les erreurs résiduelles $r_i = p_i - GT_i$. Plus la corrélation est proche de 1, plus les modèles font les mêmes erreurs."""))

cells.append(code("""target = df_val.set_index('filename').loc[FILENAMES, 'FaceOcclusion'].values
residuals = pd.DataFrame({n: PREDS[n] - target for n in PREDS})
corr = residuals.corr()
print('Corrélation des résidus (1 = redondants, 0 = orthogonaux) :')
print(corr.round(3))

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5,4))
im = ax.imshow(corr.values, vmin=0.5, vmax=1.0, cmap='RdYlGn_r')
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha='right')
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f'{corr.values[i,j]:.2f}', ha='center', va='center', fontsize=9)
plt.colorbar(im); ax.set_title('Corrélation des résidus (val, post-TTA)')
plt.tight_layout(); plt.show()
"""))

cells.append(md(r"""## 7. Recherche des poids optimaux d'ensemble

Avec 4 modèles, l'espace est de dimension 3 (le simplexe $\sum w_i = 1, w_i \ge 0$). On utilise `scipy.optimize.minimize` avec contrainte de somme = 1 et bornes $[0, 1]$ pour minimiser le `score` val.

⚠️ **Attention** : on minimise sur val → optimisme par construction. Pour avoir une estimation honnête, on fait aussi du **5-fold CV** sur les poids :
- Pour chaque fold, on optimise les poids sur 4/5 et on évalue sur 1/5
- Score moyen = estimation non-biaisée du gain de l'ensemble pondéré

Si l'écart entre `val direct` et `5-fold CV` est petit (< 5 %), on peut faire confiance aux poids optimisés."""))

cells.append(code("""from scipy.optimize import minimize

names = list(PREDS.keys())
P = np.stack([PREDS[n] for n in names])    # (M, N)
gender = df_val.set_index('filename').loc[FILENAMES, 'gender'].values
y = target

def score_from_weights(w, P=P, y=y, gender=gender):
    p = (w[:,None] * P).sum(axis=0)
    df_tmp = pd.DataFrame({'pred': p, 'target': y, 'gender': gender})
    return compute_score(df_tmp)['score']

def optimize_weights(P_arr, y_arr, g_arr, x0=None):
    M = P_arr.shape[0]
    if x0 is None:
        x0 = np.ones(M) / M
    cons = [{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}]
    bnds = [(0.0, 1.0)] * M
    res = minimize(
        lambda w: score_from_weights(w, P_arr, y_arr, g_arr),
        x0=x0, method='SLSQP', bounds=bnds, constraints=cons,
        options={'ftol': 1e-9, 'maxiter': 200},
    )
    return res

# 1) Optimisation directe sur 100% val (optimiste, mais utile pour les poids)
res_full = optimize_weights(P, y, gender)
w_full = res_full.x
print('=== Poids optimaux (val 100%) ===')
for n, w in zip(names, w_full):
    print(f'  {n:22s} : {w:.3f}')
print(f'Score ensemble pondéré (val full) : {res_full.fun:.5f}')

# 2) Référence : moyenne uniforme
w_uniform = np.ones(len(names)) / len(names)
print(f'Score ensemble uniforme           : {score_from_weights(w_uniform):.5f}')

# 3) 5-fold CV honnête : optimiser sur 4/5, évaluer sur 1/5
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_pred = np.full(len(y), np.nan)
oof_weights = []
for k, (tr, te) in enumerate(kf.split(np.arange(len(y)))):
    res_k = optimize_weights(P[:, tr], y[tr], gender[tr])
    oof_weights.append(res_k.x)
    oof_pred[te] = (res_k.x[:,None] * P[:, te]).sum(axis=0)

df_oof = pd.DataFrame({'pred': oof_pred, 'target': y, 'gender': gender})
oof_score = compute_score(df_oof)
print(f'\\n=== 5-fold CV (estimation HONNÊTE du gain ensemble pondéré) ===')
print(f'Score : {oof_score["score"]:.5f} | err_F {oof_score["err_F"]:.5f} | err_M {oof_score["err_M"]:.5f} | gap {oof_score["gap"]:.5f}')

print(f'\\nPoids moyens 5-fold (stabilité):')
w_mean = np.mean(oof_weights, axis=0)
w_std  = np.std(oof_weights, axis=0)
for n, m, s in zip(names, w_mean, w_std):
    print(f'  {n:22s} : {m:.3f} ± {s:.3f}')
"""))

cells.append(md(r"""## 8. Comparaison aux meilleures recettes Phase A

On construit un tableau qui compare :
- Phase A best : `ENS_0.4F1+0.6F2_TTA` = 0.001029 (chargé du CSV)
- 4 modèles solo (TTA)
- Ensemble uniforme 4 modèles
- Ensemble pondéré (poids optimisés sur 100 % val)
- Ensemble pondéré (5-fold CV honnête)"""))

cells.append(code("""rows = []

# Phase A best
phaseA = pd.read_csv(EXP_DIR / 'step5_phaseA_recipes.csv').iloc[0].to_dict()
rows.append({'recipe': 'PhaseA_best (ENS_0.4F1+0.6F2_TTA)', 'score': phaseA['score'],
             'err_F': phaseA['err_F'], 'err_M': phaseA['err_M'], 'gap': phaseA['gap']})

# Solos TTA
for n in names:
    df_p = pd.DataFrame({'filename': FILENAMES, 'pred': PREDS[n]}).merge(
        df_val[['filename','FaceOcclusion','gender']].rename(columns={'FaceOcclusion':'target'}), on='filename')
    s = compute_score(df_p)
    rows.append({'recipe': f'{n}_TTA', **{k: s[k] for k in ['score','err_F','err_M','gap']}})

# Uniforme 4
p_uni = (w_uniform[:,None] * P).sum(axis=0)
s_uni = compute_score(pd.DataFrame({'pred': p_uni, 'target': y, 'gender': gender}))
rows.append({'recipe': 'ENS_4uniform_TTA', **{k: s_uni[k] for k in ['score','err_F','err_M','gap']}})

# Pondéré 100% val (optimiste)
rows.append({'recipe': 'ENS_4weighted_full_TTA',
             'score': res_full.fun,
             'err_F': compute_score(pd.DataFrame({'pred':(w_full[:,None]*P).sum(0),'target':y,'gender':gender}))['err_F'],
             'err_M': compute_score(pd.DataFrame({'pred':(w_full[:,None]*P).sum(0),'target':y,'gender':gender}))['err_M'],
             'gap':   compute_score(pd.DataFrame({'pred':(w_full[:,None]*P).sum(0),'target':y,'gender':gender}))['gap']})

# Pondéré 5-fold honnête
rows.append({'recipe': 'ENS_4weighted_5foldCV_TTA',
             'score': oof_score['score'], 'err_F': oof_score['err_F'],
             'err_M': oof_score['err_M'], 'gap': oof_score['gap']})

df_phB = pd.DataFrame(rows).sort_values('score').reset_index(drop=True)
df_phB['gap_%']      = (df_phB['gap']  / df_phB['score'] * 100).round(1)
ref = phaseA['score']
df_phB['gain_vs_PhA'] = df_phB['score'] - ref
df_phB['gain_%']      = (-df_phB['gain_vs_PhA'] / ref * 100).round(2)
df_phB.to_csv(EXP_DIR / 'step6_phaseB_recipes.csv', index=False)
df_phB
"""))

cells.append(code("""fig, ax = plt.subplots(figsize=(11, max(3.5, 0.4*len(df_phB))))
ax.barh(df_phB['recipe'], df_phB['score'], color='steelblue')
ax.axvline(0.00092, color='red', ls='--', label='leaderboard 0.00092')
ax.axvline(ref, color='gray', ls=':', label='PhaseA best 0.001029')
ax.set_xlabel('val score (plus bas = mieux)')
ax.set_title('Phase B — ensembles 4 modèles')
ax.legend(loc='lower right'); ax.invert_yaxis()
plt.tight_layout(); plt.show()
"""))

cells.append(md(r"""## 9. Sauvegarde du pipeline final Phase B

On utilise les **poids moyens du 5-fold CV** comme poids finaux (plus stables que ceux optimisés sur 100 % val), normalisés à somme=1. Ça reste un compromis entre robustesse et fit val.

Le pipeline sauvegardé contient :
- liste des checkpoints (4 modèles)
- backbone associé à chaque checkpoint
- vecteur de poids
- flag TTA = True
- pas de calibration (Phase A a tranché)"""))

cells.append(code("""import pickle, re

# Poids finaux : moyenne des 5 folds, renormalisée
w_final = w_mean / w_mean.sum()
print('Poids finaux du pipeline Phase B :')
for n, w in zip(names, w_final):
    print(f'  {n:22s} : {w:.4f}')

# Score val avec ces poids (estimation optimiste, mais cohérente)
p_final = (w_final[:,None] * P).sum(axis=0)
s_final = compute_score(pd.DataFrame({'pred': p_final, 'target': y, 'gender': gender}))
print(f'\\nScore val (poids finaux appliqués sur 100% val) : {s_final["score"]:.5f}')
print(f'Estimation honnête (5-fold CV)                    : {oof_score["score"]:.5f}')

ckpt_map = {
    'F1_effb0':           ('efficientnet_b0', 'exp_F1_effb0.pt'),
    'F2_convnext':        ('convnext_tiny',   'exp_F2_convnext.pt'),
    'M3_convnext_seed1':  ('convnext_tiny',   'exp_M3_convnext_seed1.pt'),
    'M4_effb3':           ('efficientnet_b3', 'exp_M4_effb3.pt'),
}

pipeline_path = ARTIFACTS_DIR / 'pipeline_phaseB.pkl'
with open(pipeline_path, 'wb') as f:
    pickle.dump({
        'phase': 'B',
        'use_tta': True,
        'use_calib': False,
        'models': [
            {'name': n, 'backbone': ckpt_map[n][0], 'ckpt': str(ARTIFACTS_DIR / ckpt_map[n][1]),
             'weight': float(w_final[i])}
            for i, n in enumerate(names)
        ],
        'val_score_optimistic': s_final,
        'val_score_5foldCV':    oof_score,
        'note': '5-fold CV mean weights renormalized; TTA flip horizontal; no calibration (Phase A showed iso degrades)',
    }, f)
print(f'\\npipeline saved -> {pipeline_path}')

# Sauvegarde aussi les preds val pour debug
pd.DataFrame({'filename': FILENAMES, 'pred': p_final, 'target': y, 'gender': gender}).to_csv(
    EXP_DIR / 'preds_val_pipelineB.csv', index=False)
"""))

cells.append(md(r"""## 10. Conclusions Phase B

*(à remplir après exécution)*

- M3 val score : ?
- M4 val score : ?
- Diversité (corrélation min/max résidus) : ?
- Best uniforme 4 modèles : ?
- Best pondéré 5-fold CV : ?
- Gain vs Phase A best (0.001029) : ?

**Décision pour la suite** :
- Si score 5-fold CV < 0.00100 → on peut viser les ~0.00090 leaderboard. **Phase C : préparer la soumission.**
- Si score 5-fold CV ≈ 0.00100-0.00105 → ajouter encore 1-2 backbones (swin_tiny, convnext_small). **Phase B+.**
- Si score 5-fold CV > 0.00105 → revoir l'approche (capacité ? data aug ? loss ?).

**Phase C planifiée** :
1. Notebook 07 : pipeline d'inférence test (charge `pipeline_phaseB.pkl`, applique sur `test_clean.csv`)
2. Vérifier le format de sortie attendu (colonnes, ordre)
3. Submit (1ère soumission sur 10).
"""))

with open(NB_PATH, 'w') as f:
    json.dump({
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python (.venv)", "language": "python", "name": "venv"},
            "language_info": {"name": "python", "version": "3.12.3"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }, f, indent=1, ensure_ascii=False)
print(f'OK {NB_PATH} -> {len(cells)} cells')
