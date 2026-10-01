"""Smoke test du fix bias-init pour étape 4 (efficientnet_b0).

Valide en ~1-2 min sur P100 que:
  - train_loss DESCEND (et ne plateau pas à ~0.0326)
  - score val < 0.005 dès la 1re epoch (vs 0.05288 = pred=0)
  - les prédictions ne sont pas saturées à 0
"""
from __future__ import annotations
import sys, time, importlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from dc_utils.paths import IMAGE_DIR, ARTIFACTS_DIR
import dc_utils.training as T
importlib.reload(T)
from dc_utils.training import TrainConfig, train_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device = {device}, torch={torch.__version__}, gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-"}')

# --- modèle avec bias-init -----------------------------------------------------
class TimmRegressor(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool = True, bias_init: float | None = None):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=1)
        if bias_init is not None:
            head = self.backbone.get_classifier()
            with torch.no_grad():
                head.bias.fill_(bias_init)
                head.weight.mul_(0.01)
    def forward(self, x):
        return torch.sigmoid(self.backbone(x)).squeeze(-1)

# --- data ----------------------------------------------------------------------
df_train = pd.read_csv(ROOT / 'data' / 'train_split.csv')
df_val   = pd.read_csv(ROOT / 'data' / 'val_split.csv')

p0 = float(df_train['FaceOcclusion'].mean())
bias = float(np.log(p0 / (1.0 - p0)))
print(f'mean target p0 = {p0:.4f}  ->  bias_init = logit(p0) = {bias:.4f}')

mini_train = df_train.sample(5000, random_state=0).reset_index(drop=True)
mini_val   = df_val.sample(2000, random_state=0).reset_index(drop=True)

# --- run -----------------------------------------------------------------------
torch.manual_seed(42)
m = TimmRegressor('efficientnet_b0', bias_init=bias).to(device)

# sanity check: prédiction initiale doit être ~ p0
m.eval()
with torch.no_grad():
    x = torch.randn(4, 3, 224, 224, device=device)
    y = m(x)
print(f'init pred (random input) = {y.detach().cpu().numpy()}  (devrait être ~ {p0:.3f})')

cfg = TrainConfig(epochs=2, lr=1e-4, batch_size=64, use_amp=False, num_workers=4)
t0 = time.time()
hist, best = train_model(m, mini_train, mini_val, IMAGE_DIR, cfg, device, log_prefix='[smoke] ')
print(f'\nbest = {best}  |  total {time.time()-t0:.0f}s')

# critère pass/fail
print('\n--- VERDICT ---')
if best['score'] < 0.01 and hist[-1]['train_loss'] < 0.02:
    print(f"OK  fix valide : score {best['score']:.5f} < 0.01, train_loss {hist[-1]['train_loss']:.5f} < 0.02")
else:
    print(f"KO  fix insuffisant : score {best['score']:.5f}, train_loss {hist[-1]['train_loss']:.5f}")
