"""Bench v2 : cache numpy uint8 décodé (fork-safe)."""
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import pandas as pd
from torch.utils.data import DataLoader

from dc_utils.paths import IMAGE_DIR
from dc_utils.training import (OcclusionDatasetRAM, default_train_tf,
                                preload_images_decoded)

# Sample 5000 imgs (assez pour stresser les workers)
df = pd.read_csv(ROOT / 'data' / 'train_split.csv').sample(5000, random_state=0).reset_index(drop=True)

print('=== preload (decode) ===')
t0 = time.time()
preload_images_decoded(df['filename'].tolist(), IMAGE_DIR, n_threads=16)
print(f'  total : {time.time()-t0:.1f}s')

print('\n=== DataLoader avec cache uint8 array (fork-safe) ===')
ds = OcclusionDatasetRAM(df, IMAGE_DIR, default_train_tf(), has_target=True)
for nw in [0, 4, 8]:
    persistent = nw > 0
    dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=nw,
                    pin_memory=True, drop_last=True, persistent_workers=persistent)
    # warmup epoch
    for _ in dl:
        break
    # mesure
    t0 = time.time(); n = 0
    for batch in dl:
        n += batch[0].shape[0]
    dt = time.time() - t0
    print(f'  nw={nw:2d} (persistent={persistent}) | {n/dt:.0f} img/s | {dt:.1f}s')
