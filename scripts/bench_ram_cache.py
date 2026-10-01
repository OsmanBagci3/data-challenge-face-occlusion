"""Bench cache RAM vs disk."""
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import pandas as pd
from torch.utils.data import DataLoader

from dc_utils.paths import IMAGE_DIR
from dc_utils.training import (OcclusionDataset, OcclusionDatasetRAM,
                                default_train_tf, preload_image_bytes)

df = pd.read_csv(ROOT / 'data' / 'train_split.csv').sample(2048, random_state=0).reset_index(drop=True)

# On skip le test disque (déjà 16 img/s mesuré, trop lent)
print('=== preload RAM (2048 imgs) ===')
t0 = time.time()
preload_image_bytes(df['filename'].tolist(), IMAGE_DIR, n_threads=16)
print(f'  preload time : {time.time()-t0:.1f}s')

print('\n=== avec cache RAM ===')
ds_ram = OcclusionDatasetRAM(df, IMAGE_DIR, default_train_tf(), has_target=True)
for nw in [4, 8]:
    dl = DataLoader(ds_ram, batch_size=64, shuffle=False, num_workers=nw, pin_memory=True, drop_last=True)
    t0 = time.time(); n = 0
    for batch in dl:
        n += batch[0].shape[0]
    print(f'  nw={nw:2d} | {n/(time.time()-t0):.0f} img/s')
