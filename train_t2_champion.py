"""Track 2 Champion: Training MaskNet on Paired Synthetic Validation Audio.

Directly optimizes time-domain SI-SDR between:
  Input:  syntheticNoiseAudio (noisy synthetic audio)
  Target: syntheticCleanRefAudio (clean reference speech)

Architecture:
  - STFT (n_fft=512, hop=160, win=512)
  - BiGRU Mask Estimator (3 layers, hidden_size=256, dropout=0.1)
  - Masked Complex Reconstruction + iSTFT back to time-domain waveform
  - Loss: Time-domain SI-SDR Loss
"""

import os
import sys
import glob
import json
import random
import time
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Reproducibility & Config
# ---------------------------------------------------------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

SR = 16000
N_FFT = 512
HOP_LEN = 160
WIN_LEN = 512
CHUNK_SAMP = 64000  # 4 seconds chunks during training

# ---------------------------------------------------------------------------
# Metric & Loss Functions
# ---------------------------------------------------------------------------
def calc_si_sdr_np(ref, est, eps=1e-8):
    """Numpy SI-SDR for evaluation (dB, clipped [-100, 100])."""
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
    """Batched Time-Domain SI-SDR Loss (to minimize)."""
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    alpha = (est * ref).sum(dim=-1, keepdim=True) / ((ref * ref).sum(dim=-1, keepdim=True) + eps)
    t = alpha * ref
    e_res = est - t
    sdr = 10.0 * torch.log10(((t**2).sum(dim=-1) + eps) / ((e_res**2).sum(dim=-1) + eps))
    return -sdr.mean()


# ---------------------------------------------------------------------------
# MaskNet Architecture
# ---------------------------------------------------------------------------
class BiGRUMaskNet(nn.Module):
    def __init__(self, n_fft=N_FFT, hop_len=HOP_LEN, win_len=WIN_LEN, hidden_size=256, num_layers=3):
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        self.num_bins = n_fft // 2 + 1  # 257 bins
        
        self.register_buffer("window", torch.hann_window(win_len))
        
        # BiGRU on magnitude frames
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
        """
        x: (B, T) raw audio waveform at 16 kHz
        returns: (B, T) enhanced audio waveform
        """
        B, T = x.shape
        stft_c = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            win_length=self.win_len,
            window=self.window,
            return_complex=True,
        )
        mag = torch.abs(stft_c)  # (B, 257, num_frames)
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
# Dataset & DataLoader
# ---------------------------------------------------------------------------
class SyntheticPairDataset(Dataset):
    def __init__(self, pairs, chunk_samp=CHUNK_SAMP, is_train=True):
        self.pairs = pairs
        self.chunk_samp = chunk_samp
        self.is_train = is_train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_path, clean_path = self.pairs[idx]
        noisy, _ = sf.read(noisy_path, dtype="float32")
        clean, _ = sf.read(clean_path, dtype="float32")
        
        n = min(len(noisy), len(clean))
        noisy, clean = noisy[:n], clean[:n]
        
        if self.is_train:
            if n > self.chunk_samp:
                start = random.randint(0, n - self.chunk_samp)
                noisy = noisy[start : start + self.chunk_samp]
                clean = clean[start : start + self.chunk_samp]
            elif n < self.chunk_samp:
                pad_len = self.chunk_samp - n
                noisy = np.pad(noisy, (0, pad_len))
                clean = np.pad(clean, (0, pad_len))
        
        return torch.from_numpy(noisy), torch.from_numpy(clean)


# ---------------------------------------------------------------------------
# Evaluation Function
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_model(model, held_out_pairs, device):
    model.eval()
    raw_sdrs, enh_sdrs = [], []
    for noisy_path, clean_path in held_out_pairs:
        noisy, _ = sf.read(noisy_path, dtype="float32")
        clean, _ = sf.read(clean_path, dtype="float32")
        min_len = min(len(noisy), len(clean))
        noisy, clean = noisy[:min_len], clean[:min_len]
        
        inp = torch.from_numpy(noisy).unsqueeze(0).to(device)
        est, _ = model(inp)
        enh = est[0].cpu().numpy()
        
        raw_sdrs.append(calc_si_sdr_np(clean, noisy))
        enh_sdrs.append(calc_si_sdr_np(clean, enh))
        
    return float(np.mean(raw_sdrs)), float(np.mean(enh_sdrs))


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-root", type=str, default="/kaggle/working/val_data/validation")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-ckpt", type=str, default="/kaggle/working/t2_masknet_best.pt")
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using compute device: {device}")

    val_root = Path(args.val_root)
    meta_path = val_root / "validationMetadata.json"
    assert meta_path.exists(), f"validationMetadata.json not found at {meta_path}"

    with open(meta_path) as f:
        meta = json.load(f)

    synth_items = [m for m in meta if m.get("syntheticData") is True]
    print(f"[INFO] Found {len(synth_items)} synthetic metadata items.")

    pairs = []
    clean_dir = val_root / "syntheticCleanRefAudio"
    noise_dir = val_root / "syntheticNoiseAudio"

    for item in synth_items:
        fn = item["segmentFileName"]
        cp = clean_dir / fn
        np_ = noise_dir / fn
        if cp.exists() and np_.exists():
            pairs.append((str(np_), str(cp)))

    print(f"[INFO] Successfully matched {len(pairs)} (noisy, clean) audio pairs.")
    assert len(pairs) >= 500, f"Expected ~695 pairs, found {len(pairs)}"

    # Train/Validation Split (550 train, 145 held-out)
    random.Random(42).shuffle(pairs)
    train_pairs = pairs[:550]
    val_pairs = pairs[550:]
    print(f"[INFO] Train pairs: {len(train_pairs)} | Held-out validation pairs: {len(val_pairs)}")

    train_ds = SyntheticPairDataset(train_pairs, is_train=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # Initialize Model
    model = BiGRUMaskNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))

    # Baseline Evaluation before training
    raw_sdr0, enh_sdr0 = evaluate_model(model, val_pairs, device)
    print(f"\n[BASELINE] Held-out Raw SI-SDR: {raw_sdr0:+.2f} dB | Untrained Model: {enh_sdr0:+.2f} dB\n")

    best_sdr = -1e9
    print("=" * 75)
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val SI-SDR':^12} | {'Improvement':^13} | {'Time':^8}")
    print("=" * 75)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0

        for noisy_b, clean_b in train_loader:
            noisy_b = noisy_b.to(device)
            clean_b = clean_b.to(device)

            optimizer.zero_grad()
            est_b, _ = model(noisy_b)
            loss = si_sdr_loss_torch(est_b, clean_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            total_loss += float(loss.item())
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        elapsed = time.time() - t0

        raw_val_sdr, enh_val_sdr = evaluate_model(model, val_pairs, device)
        delta_sdr = enh_val_sdr - raw_val_sdr

        print(f"{epoch:^7d} | {avg_loss:^12.4f} | {enh_val_sdr:^+12.2f} | {delta_sdr:^+13.2f} | {elapsed:^7.1f}s", flush=True)

        if enh_val_sdr > best_sdr:
            best_sdr = enh_val_sdr
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_si_sdr": best_sdr,
                "delta_sdr": delta_sdr,
            }, args.save_ckpt)
            print(f"   --> Saved new best checkpoint to {args.save_ckpt} (SI-SDR: {best_sdr:.2f} dB)")

    print("=" * 75)
    print(f"[COMPLETE] Best Held-out SI-SDR: {best_sdr:+.2f} dB (Saved to {args.save_ckpt})")


if __name__ == "__main__":
    main()
