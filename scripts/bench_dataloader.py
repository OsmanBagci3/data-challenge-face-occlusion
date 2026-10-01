"""Diagnostic : mesurer le débit du DataLoader vs la GPU."""
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch
import pandas as pd
from torch.utils.data import DataLoader

from dc_utils.paths import IMAGE_DIR
from dc_utils.training import OcclusionDataset, default_train_tf

device = torch.device('cuda')
df = pd.read_csv(ROOT / 'data' / 'train_split.csv').sample(2048, random_state=0).reset_index(drop=True)

for nw in [4, 8, 12]:
    ds = OcclusionDataset(df, IMAGE_DIR, default_train_tf(), has_target=True)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=nw, pin_memory=True, drop_last=True)

    t0 = time.time()
    n_imgs = 0
    for batch in dl:
        X = batch[0]
        n_imgs += X.shape[0]
    dt = time.time() - t0
    print(f'num_workers={nw:2d} | {n_imgs} imgs in {dt:.1f}s -> {n_imgs/dt:.0f} img/s')

# Pour ref : F2 fait 89793 / 180 = ~500 img/s sur node OK
