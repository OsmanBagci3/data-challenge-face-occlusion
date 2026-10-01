"""Boucle d'entraînement réutilisable pour le data challenge.

Conçu pour comparer des configurations (sampler, augmentations, backbone, etc.)
en gardant tout le reste constant.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm.auto import tqdm

from .metric import compute_score, torch_weighted_mse


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# --- AMP helpers compatibles torch 2.2 (cuda.amp) et 2.3+ (amp) ---------------
def _make_autocast(use_amp: bool):
    """Renvoie un context manager autocast cuda compat 2.2/2.3+."""
    if hasattr(torch.amp, 'autocast'):
        try:
            return torch.amp.autocast('cuda', enabled=use_amp)
        except TypeError:
            pass
    return torch.cuda.amp.autocast(enabled=use_amp)


def _make_grad_scaler(use_amp: bool):
    """Renvoie un GradScaler cuda compat 2.2/2.3+."""
    if hasattr(torch.amp, 'GradScaler'):
        try:
            return torch.amp.GradScaler('cuda', enabled=use_amp)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def default_train_tf() -> T.Compose:
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def default_eval_tf() -> T.Compose:
    return T.Compose([
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class OcclusionDataset(Dataset):
    """Dataset images→occlusion. Renvoie (x, y, gender, filename) si has_target,
    sinon (x, filename)."""
    def __init__(self, df: pd.DataFrame, image_dir: Path, transform, has_target: bool = True):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.has_target = has_target

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.image_dir / row['filename']).convert('RGB')
        x = self.transform(img)
        if self.has_target:
            return x, np.float32(row['FaceOcclusion']), np.float32(row['gender']), row['filename']
        return x, row['filename']


# Cache global pour partager les images en RAM entre workers DataLoader.
# IMPORTANT : on stocke un *unique* numpy uint8 array (N, 224, 224, 3), pas un
# dict de bytes. Raison : Python fork() est copy-on-write au niveau pages, mais
# CPython touche le refcount de chaque objet -> avec un dict de 100k bytes, les
# 100k headers sont touchés -> COW se déclenche sur tout -> 8 workers = 8×5GB
# copiés. Avec un seul array numpy : un seul refcount, fork instantané.
_IMG_ARRAY: Optional[np.ndarray] = None        # (N, H, W, 3) uint8
_IMG_INDEX: dict[str, int] = {}                # filename -> ligne dans _IMG_ARRAY


def preload_images_decoded(
    filenames: list[str], image_dir: Path,
    img_size: int = 224, n_threads: int = 16,
) -> None:
    """Décode toutes les images en un *unique* numpy uint8 array (N, H, W, 3).

    Coûte ~1 fois le décodage de toutes les images (5-10 min pour 100k webp),
    mais après le DataLoader est ultra-rapide : pas de décodage côté worker,
    pas de Python fork bloat, le transform reçoit déjà l'image décodée.
    """
    global _IMG_ARRAY, _IMG_INDEX
    from concurrent.futures import ThreadPoolExecutor
    image_dir = Path(image_dir)
    # Si déjà chargé pour ce set de filenames -> no-op
    if _IMG_ARRAY is not None and all(fn in _IMG_INDEX for fn in filenames):
        return

    N = len(filenames)
    arr = np.empty((N, img_size, img_size, 3), dtype=np.uint8)

    def _decode(args):
        i, fn = args
        with Image.open(image_dir / fn) as im:
            im = im.convert('RGB')
            if im.size != (img_size, img_size):
                im = im.resize((img_size, img_size), Image.BILINEAR)
            arr[i] = np.asarray(im, dtype=np.uint8)
        return i

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        for _ in tqdm(ex.map(_decode, enumerate(filenames)), total=N,
                       desc=f'decode {N} imgs'):
            pass

    _IMG_ARRAY = arr
    _IMG_INDEX = {fn: i for i, fn in enumerate(filenames)}
    gb = arr.nbytes / 1e9
    print(f'  -> RAM cache: shape {arr.shape} | dtype uint8 | {gb:.2f} GB '
          f'(single ndarray, fork-safe)')


# Alias rétro-compat (vide la liste/dict mais pas besoin sur appel courant)
_BYTES_CACHE: dict = {}
def preload_image_bytes(filenames, image_dir, n_threads=16):
    """Compat : redirige vers preload_images_decoded (nouveau format optimisé)."""
    preload_images_decoded(filenames, image_dir, n_threads=n_threads)


class OcclusionDatasetRAM(Dataset):
    """Lit les images directement depuis _IMG_ARRAY (numpy uint8 partagé).

    Le transform reçoit une PIL Image construite depuis la mémoire (zéro décode webp).
    """
    def __init__(self, df: pd.DataFrame, image_dir: Path, transform, has_target: bool = True):
        self.filenames = df['filename'].to_numpy()                 # array contigu
        self.targets   = df['FaceOcclusion'].to_numpy(dtype=np.float32) if has_target else None
        self.genders   = df['gender'].to_numpy(dtype=np.float32) if has_target else None
        self.transform = transform
        self.has_target = has_target
        if _IMG_ARRAY is None:
            raise RuntimeError('RAM cache vide -> appelle preload_images_decoded() d\'abord')
        # Vérifie quelques filenames
        for fn in self.filenames[:5]:
            if fn not in _IMG_INDEX:
                raise RuntimeError(f'cache miss: {fn}')

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fn = self.filenames[idx]
        i  = _IMG_INDEX[fn]
        # _IMG_ARRAY[i] est une vue (224, 224, 3) uint8 -> on construit une PIL Image
        # qui partage le buffer. Le transform fera ToTensor() qui copie en float32.
        img = Image.fromarray(_IMG_ARRAY[i])
        x = self.transform(img)
        if self.has_target:
            return x, self.targets[idx], self.genders[idx], fn
        return x, fn


def make_weighted_sampler(df: pd.DataFrame, mode: str = "metric") -> torch.utils.data.WeightedRandomSampler:
    """Sampler dont les poids tirent plus souvent les exemples qui pèsent dans la métrique.

    - mode='metric' : poids = w_i = 1/30 + GT_i. Cohérent avec la loss pondérée.
    - mode='gender_balanced' : poids = 1/freq(gender). Force chaque batch à ~50/50 H/F.
    - mode='gender_x_bucket' : poids = 1/freq(gender, occ_bin). Équilibre genre×bucket.
    """
    if mode == "metric":
        w = (1.0 / 30.0 + df["FaceOcclusion"].values).astype(np.float64)
    elif mode == "gender_balanced":
        freq = df["gender"].value_counts().to_dict()
        w = df["gender"].map(lambda g: 1.0 / freq[g]).values.astype(np.float64)
    elif mode == "gender_x_bucket":
        key = df["gender"].astype(str) + "|" + df["occ_bin"].astype(str)
        freq = key.value_counts().to_dict()
        w = key.map(lambda k: 1.0 / freq[k]).values.astype(np.float64)
    else:
        raise ValueError(f"unknown mode: {mode}")
    w = torch.from_numpy(w)
    return torch.utils.data.WeightedRandomSampler(w, num_samples=len(df), replacement=True)


@dataclass
class TrainConfig:
    epochs: int = 8
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    early_stop_patience: int = 0   # 0 = désactivé
    use_amp: bool = True
    sampler_mode: Optional[str] = None  # None, "metric", "gender_balanced", "gender_x_bucket"
    train_tf: Optional[Callable] = None
    eval_tf: Optional[Callable] = None
    cosine: bool = True
    cache_in_ram: bool = False  # si True : pré-charge bytes en RAM, utilise OcclusionDatasetRAM


@torch.inference_mode()
def evaluate(model, dl, device, use_amp=True) -> tuple[dict, pd.DataFrame]:
    model.eval()
    preds, tgts, gnds, fns = [], [], [], []
    ctx = _make_autocast(use_amp)
    for X, y, g, fn in dl:
        X = X.to(device, non_blocking=True)
        with ctx:
            p = model(X).float().cpu().numpy()
        preds.append(p); tgts.append(y.numpy()); gnds.append(g.numpy()); fns.extend(fn)
    df = pd.DataFrame({
        'filename': fns,
        'pred': np.concatenate(preds),
        'target': np.concatenate(tgts),
        'gender': np.concatenate(gnds),
    })
    return compute_score(df), df


def train_model(
    model: nn.Module,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    image_dir: Path,
    cfg: TrainConfig,
    device: torch.device,
    save_path: Optional[Path] = None,
    log_prefix: str = "",
) -> tuple[pd.DataFrame, dict]:
    """Entraîne `model`, retourne (history_df, best_val_metrics).

    Sauvegarde le meilleur checkpoint (selon `score` val) à `save_path` si fourni.
    """
    train_tf = cfg.train_tf or default_train_tf()
    eval_tf  = cfg.eval_tf  or default_eval_tf()

    if cfg.cache_in_ram:
        all_files = list(df_train['filename']) + list(df_val['filename'])
        preload_images_decoded(all_files, image_dir)
        ds_train = OcclusionDatasetRAM(df_train, image_dir, train_tf, has_target=True)
        ds_val   = OcclusionDatasetRAM(df_val,   image_dir, eval_tf,  has_target=True)
    else:
        ds_train = OcclusionDataset(df_train, image_dir, train_tf, has_target=True)
        ds_val   = OcclusionDataset(df_val,   image_dir, eval_tf,  has_target=True)

    if cfg.sampler_mode is None:
        dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=cfg.num_workers > 0)
    else:
        sampler = make_weighted_sampler(df_train, cfg.sampler_mode)
        dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, sampler=sampler,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=cfg.num_workers > 0)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True,
                        persistent_workers=cfg.num_workers > 0)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs) if cfg.cosine else None
    scaler = _make_grad_scaler(cfg.use_amp)

    best_score = float('inf')
    best_metrics: dict = {}
    history = []
    no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        pbar = tqdm(dl_train, desc=f'{log_prefix}ep{epoch}', leave=False)
        for X, y, g, fn in pbar:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with _make_autocast(cfg.use_amp):
                pred = model(X)
                loss = torch_weighted_mse(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            losses.append(loss.item())
            pbar.set_postfix(loss=f'{loss.item():.5f}')
        if sched is not None:
            sched.step()
        train_loss = float(np.mean(losses))
        val_metrics, _ = evaluate(model, dl_val, device, cfg.use_amp)
        dt = time.time() - t0
        history.append({'epoch': epoch, 'train_loss': train_loss, **val_metrics, 'sec': dt})

        improved = val_metrics['score'] < best_score
        flag = ' *' if improved else ''
        print(f'{log_prefix}ep{epoch:2d} | loss {train_loss:.5f} | '
              f'score {val_metrics["score"]:.5f} | err_F {val_metrics["err_F"]:.5f} | '
              f'err_M {val_metrics["err_M"]:.5f} | gap {val_metrics["gap"]:.5f} | {dt:.0f}s{flag}')

        if improved:
            best_score = val_metrics['score']
            best_metrics = {**val_metrics, 'epoch': epoch}
            no_improve = 0
            if save_path is not None:
                torch.save({'model': model.state_dict(), 'epoch': epoch, 'val': val_metrics}, save_path)
        else:
            no_improve += 1
            if cfg.early_stop_patience > 0 and no_improve >= cfg.early_stop_patience:
                print(f'{log_prefix}early stop (patience {cfg.early_stop_patience}) at epoch {epoch}')
                break

    return pd.DataFrame(history), best_metrics
