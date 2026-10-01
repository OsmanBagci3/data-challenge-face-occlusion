"""Phase E : MLP stacker robuste qui minimise directement la métrique.

Loss = (err_F + err_M)/2 + |err_F - err_M|, avec err_g pondérée w = 1/30 + GT.
Tout différentiable -> backprop direct.

Robustesse :
- 5 seeds par fold -> moyenne des prédictions
- early stopping avec patience
- standardisation features

Sauve :
- experiments/phaseE_mlp_stacker_oof.npz : OOF prédictions sur val
- artifacts/mlp_stacker_phaseE/fold{k}_seed{s}.pt : 25 checkpoints
- artifacts/mlp_stacker_phaseE/feat_scaler.npz : mu, sd
- artifacts/pipeline_phaseE.pkl : pipeline complet pour inference
"""
from __future__ import annotations
import sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dc_utils.metric import compute_score
from sklearn.model_selection import KFold

# ----------------------------- Data -----------------------------
arc_oof = np.load(ROOT / 'experiments/phaseC_oof.npz')
arc_val = np.load(ROOT / 'experiments/phaseC_preds_val.npz')
val = pd.read_csv(ROOT / 'data/val_split.csv')
fns = list(arc_val['filenames'])
val = val.set_index('filename').loc[fns].reset_index()
with open(ROOT / 'artifacts/pipeline_phaseC.pkl', 'rb') as f:
    pipe = pickle.load(f)
names = pipe['model_names']

P_val = np.stack([arc_val[n] for n in names])
y = val['FaceOcclusion'].values.astype(np.float32)
g = val['gender'].values.astype(np.float32)

feats = {f'pred_{names[i]}': P_val[i] for i in range(len(names))}
feats['gender']   = g
feats['p_mean']   = P_val.mean(0)
feats['p_std']    = P_val.std(0)
feats['p_min']    = P_val.min(0)
feats['p_max']    = P_val.max(0)
feats['p_range']  = feats['p_max'] - feats['p_min']
feats['p_median'] = np.median(P_val, 0)
feature_columns = pipe['stacking']['feature_columns']
X = pd.DataFrame(feats)[feature_columns].values.astype(np.float32)

mu = X.mean(0, keepdims=True).astype(np.float32)
sd = (X.std(0, keepdims=True) + 1e-6).astype(np.float32)
Xn = ((X - mu) / sd).astype(np.float32)
print(f'X: {X.shape} | features: {feature_columns}')

# ----------------------------- Loss -----------------------------
def metric_loss(pred, target, gender):
    w = 1.0/30.0 + target
    se = (pred - target)**2 * w
    mF = (gender < 0.5).float()
    mM = 1.0 - mF
    eF = (se * mF).sum() / (w * mF).sum().clamp_min(1e-9)
    eM = (se * mM).sum() / (w * mM).sum().clamp_min(1e-9)
    return 0.5*(eF + eM) + (eF - eM).abs()

# ----------------------------- Model -----------------------------
class MLPStacker(nn.Module):
    def __init__(self, in_dim, hidden=64, drop=0.1, bias_init=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, 1),
        )
        with torch.no_grad():
            self.net[-1].bias.fill_(bias_init)
            self.net[-1].weight.mul_(0.01)

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


# ----------------------------- Train one fold/seed -----------------------------
def train_one(Xtr, ytr, gtr, Xva, yva, gva, seed, hidden=64, drop=0.1,
              lr=2e-3, wd=1e-4, epochs=400, bs=512, patience=50):
    torch.manual_seed(seed); np.random.seed(seed)
    p0 = float(np.clip(ytr.mean(), 1e-3, 1-1e-3))
    bias0 = float(np.log(p0/(1-p0)))
    m = MLPStacker(Xtr.shape[1], hidden=hidden, drop=drop, bias_init=bias0).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    gtr_t = torch.from_numpy(gtr).to(device)
    Xva_t = torch.from_numpy(Xva).to(device)
    yva_t = torch.from_numpy(yva).to(device)
    gva_t = torch.from_numpy(gva).to(device)

    best, bad, best_pred, best_ep = float('inf'), 0, None, 0
    best_state = None
    n = len(Xtr)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            loss = metric_loss(m(Xtr_t[idx]), ytr_t[idx], gtr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        m.eval()
        with torch.no_grad():
            pv = m(Xva_t)
            sc = metric_loss(pv, yva_t, gva_t).item()
        if sc < best:
            best = sc; best_pred = pv.cpu().numpy().copy(); best_ep = ep; bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    return best_pred, best, best_ep, best_state


# ----------------------------- 5-fold × N seeds -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

ART_DIR = ROOT / 'artifacts/mlp_stacker_phaseE'
ART_DIR.mkdir(exist_ok=True)
np.savez(ART_DIR / 'feat_scaler.npz', mu=mu, sd=sd, feature_columns=np.array(feature_columns))

kf = KFold(5, shuffle=True, random_state=42)
oof_mean = np.zeros(len(y), dtype=np.float32)
SEEDS = [0, 1, 2, 3, 4]

fold_scores = []
ckpt_files = []
for k, (tr, te) in enumerate(kf.split(Xn)):
    fold_preds = []
    for s in SEEDS:
        pred, sc, ep, state = train_one(
            Xn[tr], y[tr], g[tr], Xn[te], y[te], g[te],
            seed=s, hidden=64, drop=0.1, lr=2e-3, wd=1e-4,
            epochs=400, bs=512, patience=50,
        )
        fold_preds.append(pred)
        # Save state
        ckpt_path = ART_DIR / f'fold{k}_seed{s}.pt'
        torch.save({'state': state, 'best_score': sc, 'best_ep': ep}, ckpt_path)
        ckpt_files.append(str(ckpt_path))
    fold_avg = np.mean(fold_preds, axis=0)
    oof_mean[te] = fold_avg
    sc_f = compute_score(pd.DataFrame({
        'pred': fold_avg, 'target': y[te], 'gender': g[te]
    }))
    fold_scores.append(sc_f['score'])
    print(f'  fold {k}: avg-of-{len(SEEDS)} score={sc_f["score"]:.6f} '
          f'(seed scores: indiv ~{np.median([s for s in [sc_f["score"]]]):.6f})')

oof_mean = np.clip(oof_mean, 0, 1)
s_mlp = compute_score(pd.DataFrame({'pred': oof_mean, 'target': y, 'gender': g}))
print()
print('=== MLP STACKER ROBUSTE (5-fold × {} seeds) ==='.format(len(SEEDS)))
print(f'  score = {s_mlp["score"]:.6f}')
print(f'  err_F = {s_mlp["err_F"]:.6f}')
print(f'  err_M = {s_mlp["err_M"]:.6f}')
print(f'  gap   = {s_mlp["gap"]:.6f}')
alpha_old = pipe['blend_alpha']
oof_old = np.clip(alpha_old*arc_oof['oof_pred_stk'] + (1-alpha_old)*arc_oof['oof_pred'], 0, 1)
base = compute_score(pd.DataFrame({'pred': oof_old, 'target': y, 'gender': g}))
print(f'  base blend C : {base["score"]:.6f}  (Δ {(s_mlp["score"]-base["score"])/base["score"]*100:+.1f}%)')

# ----------------------------- Search 3-blend -----------------------------
print()
print('--- search 3-blend (mlp, lgbm, weighted) ---')
oof_w = arc_oof['oof_pred']; oof_stk = arc_oof['oof_pred_stk']
best_3, best_cfg = float('inf'), None
for am in np.linspace(0, 1, 41):
    for ast in np.linspace(0, 1-am, 41):
        aw = 1 - am - ast
        if aw < -1e-9: continue
        pred = np.clip(am*oof_mean + ast*oof_stk + aw*oof_w, 0, 1)
        sc = compute_score(pd.DataFrame({'pred': pred, 'target': y, 'gender': g}))['score']
        if sc < best_3:
            best_3, best_cfg = sc, (am, ast, aw)
print(f'BEST 3-blend: am={best_cfg[0]:.3f}, ast={best_cfg[1]:.3f}, aw={best_cfg[2]:.3f} -> {best_3:.6f}')

# ----------------------------- Save artefacts -----------------------------
np.savez(ROOT / 'experiments/phaseE_mlp_stacker_oof.npz',
         oof_mlp=oof_mean,
         blend_cfg=np.array(best_cfg),
         best_blend_score=np.array([best_3]))

pipeline_E = dict(pipe)
pipeline_E['phase'] = 'E'
pipeline_E['mlp_stacker'] = {
    'feat_scaler_file': str(ART_DIR / 'feat_scaler.npz'),
    'feature_columns': feature_columns,
    'ckpt_files': ckpt_files,
    'arch': {'hidden': 64, 'drop': 0.1, 'in_dim': Xn.shape[1]},
    'n_folds': 5,
    'n_seeds': len(SEEDS),
}
pipeline_E['blend_3'] = {
    'am_mlp': float(best_cfg[0]),
    'ast_lgbm': float(best_cfg[1]),
    'aw_weighted': float(best_cfg[2]),
}
pipeline_E['val_score_5foldCV_mlp'] = s_mlp
pipeline_E['val_score_5foldCV_blend3'] = {
    'score': float(best_3),
}
pipeline_E['note'] = ('Phase E: MLP stacker minimisant directement la métrique '
                     '+ blend 3-way (mlp/lgbm/weighted). '
                     f'CV: blend3 = {best_3:.6f} (was C={base["score"]:.6f}, Δ {(best_3-base["score"])/base["score"]*100:+.1f}%)')
with open(ROOT / 'artifacts/pipeline_phaseE.pkl', 'wb') as f:
    pickle.dump(pipeline_E, f)
print()
print(f'saved {ROOT / "artifacts/pipeline_phaseE.pkl"}')
print(f'saved {ROOT / "experiments/phaseE_mlp_stacker_oof.npz"}')
print(f'saved {len(ckpt_files)} MLP checkpoints in {ART_DIR}')
