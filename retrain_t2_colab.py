"""Retrain the Synapse Track-2 BiGRU MaskNet on Colab (no network dependencies beyond Colab base).

Run as a single Colab notebook, button order:
  1. Mount Google Drive (code below does it, or use the Drive mount icon).
  2. Put the official validation data (Codabench -> Get Started -> Files -> input_data)
     as a zip on Drive, e.g.  MyDrive/indoml/validation.zip
  3. Run this whole file as a cell (or %run retrain_t2_colab.py).
  4. After training, download  MyDrive/indoml/t2_masknet_best.pt  locally.

Produces a checkpoint whose dict keys match the packaged main.py:
  {"epoch": ..., "model_state_dict": ..., "val_si_sdr": ..., "delta_sdr": ...}
"""

from __future__ import annotations

import glob
import json
import os
import random
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Google Drive mount
# ---------------------------------------------------------------------------
from google.colab import drive

drive.mount("/content/drive")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SR = 16000
N_FFT = 512
HOP_LEN = 160
WIN_LEN = 512
CHUNK_SAMP = 64000  # 4 s chunks during training

DRIVE_ROOT = Path("/content/drive/MyDrive/indoml")
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)

AUTO_FIND_VALIDATION = True          # search Drive for validation zip / folder
VAL_ZIP = DRIVE_ROOT / "validation.zip"

CKPT_OUT = DRIVE_ROOT / "t2_masknet_best.pt"

EPOCHS = 25
BATCH_SIZE = 16
LR = 1e-3


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model (verbatim from train_t2_champion.py)
# ---------------------------------------------------------------------------
class BiGRUMaskNet(nn.Module):
    def __init__(self, n_fft=N_FFT, hop_len=HOP_LEN, win_len=WIN_LEN, hidden_size=256, num_layers=3):
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        self.num_bins = n_fft // 2 + 1

        self.register_buffer("window", torch.hann_window(win_len))

        self.in_proj = nn.Linear(self.num_bins, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, self.num_bins),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, T = x.shape
        stft_c = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            win_length=self.win_len,
            window=self.window,
            return_complex=True,
        )
        mag = torch.abs(stft_c)
        phase = torch.angle(stft_c)

        feat = mag.permute(0, 2, 1)
        h = self.in_proj(feat)
        gru_out, _ = self.gru(h)
        mask = self.out_proj(gru_out)
        mask = mask.permute(0, 2, 1)

        est_mag = mag * mask
        est_c = torch.polar(est_mag, phase)

        est_wav = torch.istft(
            est_c,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            win_length=self.win_len,
            window=self.window,
            length=T,
        )
        return est_wav, mask


# ---------------------------------------------------------------------------
# Loss / metric
# ---------------------------------------------------------------------------
def calc_si_sdr_np(ref, est, eps=1e-8):
    ref = np.asarray(ref, dtype=np.float64)
    est = np.asarray(est, dtype=np.float64)
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    t = alpha * ref
    res = est - t
    val = 10.0 * np.log10((np.sum(t**2) + eps) / (np.sum(res**2) + eps))
    return float(np.clip(val, -100.0, 100.0))


def si_sdr_loss_torch(est, ref, eps=1e-8):
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    alpha = (est * ref).sum(dim=-1, keepdim=True) / ((ref * ref).sum(dim=-1, keepdim=True) + eps)
    t = alpha * ref
    e_res = est - t
    sdr = 10.0 * torch.log10(((t**2).sum(dim=-1) + eps) / ((e_res**2).sum(dim=-1) + eps))
    return -sdr.mean()


class SyntheticPairDataset(Dataset):
    def __init__(self, pairs, chunk_samp=CHUNK_SAMP, is_train=True):
        self.pairs = pairs
        self.chunk_samp = chunk_samp
        self.is_train = is_train

    def __getitem__(self, idx):
        noisy_path, clean_path = self.pairs[idx]
        noisy, _ = sf.read(noisy_path, dtype="float32")
        clean, _ = sf.read(clean_path, dtype="float32")
        n = min(len(noisy), len(clean))
        noisy, clean = noisy[:n], clean[:n]
        if self.is_train:
            if n > self.chunk_samp:
                start = random.randint(0, n - self.chunk_samp)
                noisy = noisy[start: start + self.chunk_samp]
                clean = clean[start: start + self.chunk_samp]
            elif n < self.chunk_samp:
                pad = self.chunk_samp - n
                noisy = np.pad(noisy, (0, pad))
                clean = np.pad(clean, (0, pad))
        return torch.from_numpy(noisy), torch.from_numpy(clean)

    def __len__(self):
        return len(self.pairs)


@torch.no_grad()
def evaluate_model(model, held_out_pairs, device):
    model.eval()
    raw_sdrs, enh_sdrs = [], []
    for noisy_path, clean_path in held_out_pairs:
        noisy, _ = sf.read(noisy_path, dtype="float32")
        clean, _ = sf.read(clean_path, dtype="float32")
        n = min(len(noisy), len(clean))
        noisy, clean = noisy[:n], clean[:n]
        inp = torch.from_numpy(noisy).unsqueeze(0).to(device)
        est, _ = model(inp)
        enh = est[0].cpu().numpy()
        raw_sdrs.append(calc_si_sdr_np(clean, noisy))
        enh_sdrs.append(calc_si_sdr_np(clean, enh))
    return float(np.mean(raw_sdrs)), float(np.mean(enh_sdrs))


# ---------------------------------------------------------------------------
# Locate validation data (auto-search Drive)  -> returns metadata folder path
# ---------------------------------------------------------------------------
def find_validation():
    meta_name = "validationMetadata.json"

    # 1) any zip you curl'd into /content (e.g. /content/validation.zip, /content/input_data.zip)
    for zp in sorted(Path("/content").glob("*.zip")):
        print(f"[INFO] Scanning {zp} for metadata ...")
        probe = Path("/content/_valprobe")
        with zipfile.ZipFile(zp) as z:
            names = z.namelist()
        if any(n.endswith(meta_name) for n in names):
            print(f"[INFO] Unzipping {zp} ...")
            with zipfile.ZipFile(zp) as z:
                z.extractall(probe)
                for mp in probe.rglob(meta_name):
                    print(f"[INFO] Metadata at {mp.parent}")
                    return mp.parent

    # 2) explicit Drive zip
    if VAL_ZIP.exists():
        print(f"[INFO] Unzipping {VAL_ZIP} ...")
        dst = Path("/content/val_data")
        with zipfile.ZipFile(VAL_ZIP) as z:
            z.extractall(dst)
        for mp in dst.rglob(meta_name):
            print(f"[INFO] Found metadata at {mp.parent}")
            return mp.parent
        raise FileNotFoundError(
            f"{meta_name} not inside {VAL_ZIP}. Check the zip content / download the right file."
        )

    # 3) already-extracted folder anywhere on Drive or /content
    hits = list(DRIVE_ROOT.rglob(meta_name))
    if not hits and "/content/drive" in str(DRIVE_ROOT):
        hits = list(Path("/content/drive/MyDrive").rglob(meta_name))
    if not hits:
        hits = glob.glob("/content/**/validationMetadata.json", recursive=True)
    if hits:
        p = Path(hits[0]).parent
        print(f"[INFO] Found metadata at {p}")
        return p
    raise FileNotFoundError(
        f"Could not locate {meta_name} anywhere. "
        "Curl the official validation zip into /content (e.g. /content/validation.zip) "
        "or place it at MyDrive/indoml/validation.zip"
    )


def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    val_root = find_validation()
    meta_path = val_root / "validationMetadata.json"
    with open(meta_path) as f:
        meta = json.load(f)

    synth = [m for m in meta if m.get("syntheticData") is True]
    print(f"[INFO] Synthetic items in metadata: {len(synth)}")

    pairs = []
    clean_dir = val_root / "syntheticCleanRefAudio"
    noise_dir = val_root / "syntheticNoiseAudio"
    for item in synth:
        fn = item["segmentFileName"]
        cp, np_ = clean_dir / fn, noise_dir / fn
        if cp.exists() and np_.exists():
            pairs.append((str(np_), str(cp)))

    print(f"[INFO] Matched (noisy, clean) pairs: {len(pairs)}")
    assert len(pairs) >= 500, f"Expected ~695 pairs, found {len(pairs)}. Data looks wrong/truncated."

    random.Random(42).shuffle(pairs)
    train_pairs, val_pairs = pairs[:550], pairs[550:]
    print(f"[INFO] Train: {len(train_pairs)} | Held-out val: {len(val_pairs)}")

    loader = DataLoader(SyntheticPairDataset(train_pairs, is_train=True),
                        batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    model = BiGRUMaskNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * len(loader))

    raw0, enh0 = evaluate_model(model, val_pairs, device)
    print(f"\n[BASELINE] Raw SI-SDR: {raw0:+.2f} dB | Untrained model: {enh0:+.2f} dB\n")

    best_sdr, best_epoch = -1e9, 0
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        total, nb = 0.0, 0
        for noisy_b, clean_b in loader:
            noisy_b, clean_b = noisy_b.to(device), clean_b.to(device)
            opt.zero_grad()
            est_b, _ = model(noisy_b)
            loss = si_sdr_loss_torch(est_b, clean_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            sched.step()
            total += float(loss.item())
            nb += 1

        raw_v, enh_v = evaluate_model(model, val_pairs, device)
        delta = enh_v - raw_v
        print(f"Epoch {epoch:3d} | loss {total/max(1,nb):.4f} | val SI-SDR {enh_v:+.2f} dB "
              f"({delta:+.2f}) | {time.time()-t0:.1f}s", flush=True)

        if enh_v > best_sdr:
            best_sdr, best_epoch = enh_v, epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_si_sdr": best_sdr,
                "delta_sdr": delta,
            }, CKPT_OUT)
            print(f"   --> saved {CKPT_OUT} (SI-SDR {best_sdr:.2f} dB)")

    print(f"\n[DONE] Best val SI-SDR {best_sdr:+.2f} dB at epoch {best_epoch}")
    print(f"[DONE] Checkpoint on Drive: {CKPT_OUT}")
    print("Download it: from the Files panel  /content/drive/MyDrive/indoml/t2_masknet_best.pt")


if __name__ == "__main__":
    main()