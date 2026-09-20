"""Complex Ratio MaskNet (cRM) for IndoML 2026 Track 2 Speech Enhancement.

Architecture:
- Front-End: Complex STFT (n_fft=512, hop=160, win=512).
- Input Representation: Concatenated Real, Imaginary, and Magnitude spectrograms (3 * 257 = 771 features).
- Core Separator: 4-Layer BiGRU with LayerNorm and Residual Connections (hidden_size=384).
- Dual Output Heads:
    * Real Mask M_r in [-2.0, 2.0]
    * Imaginary Mask M_i in [-2.0, 2.0]
- Complex Spectral Multiplication:
    S_r = M_r * X_r - M_i * X_i
    S_i = M_r * X_i + M_i * X_r
  Enables active phase rotation and noise cancellation in the complex plane.
- Reconstruction: iSTFT back to time-domain waveform.
- Dual-Domain Loss: Time-Domain SI-SDR Loss + Multi-Resolution STFT Loss.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Signal Processing Parameters
# ---------------------------------------------------------------------------
SR = 16000
N_FFT = 512
HOP_LEN = 160
WIN_LEN = 512
CHUNK_SAMP = 64000  # 4-second slices during training


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Multi-Resolution STFT Loss
# ---------------------------------------------------------------------------
class MultiResolutionSTFTLoss(nn.Module):
    def __init__(
        self,
        fft_sizes: list[int] = [512, 1024, 2048],
        hop_sizes: list[int] = [128, 256, 512],
        win_lengths: list[int] = [512, 1024, 2048],
    ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for n_fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            w = torch.hann_window(win, device=est.device)
            est_stft = torch.stft(est, n_fft, hop, win, window=w, return_complex=True)
            ref_stft = torch.stft(ref, n_fft, hop, win, window=w, return_complex=True)
            
            est_mag = torch.abs(est_stft) + 1e-7
            ref_mag = torch.abs(ref_stft) + 1e-7
            
            # Spectral Convergence Loss
            sc_loss = torch.norm(ref_mag - est_mag, p="fro") / (torch.norm(ref_mag, p="fro") + 1e-7)
            # Log Magnitude STFT Loss
            log_loss = F.l1_loss(torch.log(est_mag), torch.log(ref_mag))
            
            total_loss += (sc_loss + log_loss)
        return total_loss / len(self.fft_sizes)


def calc_si_sdr_np(ref: np.ndarray, est: np.ndarray, eps: float = 1e-8) -> float:
    """Numpy SI-SDR calculation clipped to [-100, 100] dB."""
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


def si_sdr_loss_torch(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Batched Time-Domain SI-SDR loss (negated to minimize)."""
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    alpha = (est * ref).sum(dim=-1, keepdim=True) / ((ref * ref).sum(dim=-1, keepdim=True) + eps)
    t = alpha * ref
    e_res = est - t
    sdr = 10.0 * torch.log10(((t**2).sum(dim=-1) + eps) / ((e_res**2).sum(dim=-1) + eps))
    return -sdr.mean()


# ---------------------------------------------------------------------------
# Complex Ratio MaskNet Model
# ---------------------------------------------------------------------------
class ComplexMaskNet(nn.Module):
    def __init__(
        self,
        n_fft: int = N_FFT,
        hop_len: int = HOP_LEN,
        win_len: int = WIN_LEN,
        hidden_size: int = 384,
        num_layers: int = 4,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        self.num_bins = n_fft // 2 + 1  # 257 bins
        
        self.register_buffer("window", torch.hann_window(win_len))
        
        # Real, Imag, and Mag features: 257 * 3 = 771
        in_dim = self.num_bins * 3
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.PReLU(),
        )
        
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.15 if num_layers > 1 else 0.0,
        )
        
        # Complex Mask Heads
        self.real_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, self.num_bins),
            nn.Tanh(),
        )
        self.imag_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, self.num_bins),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, T) raw audio at 16 kHz
        returns: (B, T) enhanced audio, (B, F, T) real mask, (B, F, T) imag mask
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
        xr = stft_c.real  # (B, F, num_frames)
        xi = stft_c.imag  # (B, F, num_frames)
        xmag = torch.abs(stft_c)

        # Concatenate Real, Imag, Mag along frequency dimension
        # (B, 3*F, num_frames) -> permute to (B, num_frames, 3*F)
        feat = torch.cat([xr, xi, xmag], dim=1).permute(0, 2, 1)
        h = self.in_proj(feat)
        gru_out, _ = self.gru(h)

        # Scale tanh to [-2.0, 2.0]
        mr = self.real_head(gru_out).permute(0, 2, 1) * 2.0
        mi = self.imag_head(gru_out).permute(0, 2, 1) * 2.0

        # Complex ratio multiplication:
        # S_real = M_r * X_r - M_i * X_i
        # S_imag = M_r * X_i + M_i * X_r
        sr = mr * xr - mi * xi
        si = mr * xi + mi * xr
        s_c = torch.complex(sr, si)

        # Invert to time domain
        est_wav = torch.istft(
            s_c,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            win_length=self.win_len,
            window=self.window,
            length=T,
        )
        return est_wav, mr, mi


# ---------------------------------------------------------------------------
# Dataset & Augmentation
# ---------------------------------------------------------------------------
class ComplexAudioPairDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]], chunk_samp: int = CHUNK_SAMP, is_train: bool = True):
        self.pairs = pairs
        self.chunk_samp = chunk_samp
        self.is_train = is_train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        noisy_path, clean_path = self.pairs[idx]
        noisy, _ = sf.read(noisy_path, dtype="float32")
        clean, _ = sf.read(clean_path, dtype="float32")

        n = min(len(noisy), len(clean))
        noisy, clean = noisy[:n], clean[:n]

        if self.is_train:
            # Data Augmentation: dynamic SNR re-weighting
            if random.random() < 0.35:
                noise_part = noisy - clean
                scale = random.uniform(0.5, 1.5)
                noisy = clean + scale * noise_part

            # Random Gain Jitter
            gain = random.uniform(0.8, 1.2)
            noisy = noisy * gain
            clean = clean * gain

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
# Model Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_model(model: nn.Module, held_out_pairs: list[tuple[str, str]], device: torch.device) -> tuple[float, float]:
    model.eval()
    raw_sdrs, enh_sdrs = [], []
    for noisy_path, clean_path in held_out_pairs:
        noisy, _ = sf.read(noisy_path, dtype="float32")
        clean, _ = sf.read(clean_path, dtype="float32")
        min_len = min(len(noisy), len(clean))
        noisy, clean = noisy[:min_len], clean[:min_len]

        inp = torch.from_numpy(noisy).unsqueeze(0).to(device)
        est, _, _ = model(inp)
        enh = est[0].cpu().numpy()

        raw_sdrs.append(calc_si_sdr_np(clean, noisy))
        enh_sdrs.append(calc_si_sdr_np(clean, enh))

    return float(np.mean(raw_sdrs)), float(np.mean(enh_sdrs))


# ---------------------------------------------------------------------------
# Training Orchestration
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Complex Ratio MaskNet Training")
    parser.add_argument("--val-root", type=str, default="/kaggle/working/val_data/validation")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--stft-weight", type=float, default=0.20)
    parser.add_argument("--save-ckpt", type=str, default="/kaggle/working/t2_complex_masknet_best.pt")
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
    clean_dir = val_root / "syntheticCleanRefAudio"
    noise_dir = val_root / "syntheticNoiseAudio"

    pairs = []
    for item in synth_items:
        fn = item["segmentFileName"]
        cp = clean_dir / fn
        np_ = noise_dir / fn
        if cp.exists() and np_.exists():
            pairs.append((str(np_), str(cp)))

    print(f"[INFO] Successfully matched {len(pairs)} (noisy, clean) audio pairs.")

    random.Random(42).shuffle(pairs)
    train_pairs = pairs[:550]
    val_pairs = pairs[550:]
    print(f"[INFO] Train pairs: {len(train_pairs)} | Held-out validation pairs: {len(val_pairs)}")

    train_ds = ComplexAudioPairDataset(train_pairs, is_train=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    model = ComplexMaskNet(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Initialized ComplexMaskNet ({total_params / 1e6:.2f}M trainable parameters).")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader), eta_min=1e-6)
    mr_stft_loss_fn = MultiResolutionSTFTLoss().to(device)

    raw_val0, enh_val0 = evaluate_model(model, val_pairs, device)
    print(f"\n[BASELINE] Held-out Raw SI-SDR: {raw_val0:+.2f} dB | Untrained Model: {enh_val0:+.2f} dB\n")

    best_sdr = -1e9
    print("=" * 85)
    print(f"{'Epoch':^7} | {'Total Loss':^12} | {'SI-SDR Loss':^12} | {'Val SI-SDR':^12} | {'Delta':^9} | {'Time':^8}")
    print("=" * 85)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tot_loss_accum = 0.0
        sdr_loss_accum = 0.0
        n_batches = 0

        for noisy_b, clean_b in train_loader:
            noisy_b = noisy_b.to(device)
            clean_b = clean_b.to(device)

            optimizer.zero_grad()
            est_b, _, _ = model(noisy_b)

            loss_sdr = si_sdr_loss_torch(est_b, clean_b)
            loss_stft = mr_stft_loss_fn(est_b, clean_b)
            loss = loss_sdr + args.stft_weight * loss_stft

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            tot_loss_accum += float(loss.item())
            sdr_loss_accum += float(loss_sdr.item())
            n_batches += 1

        avg_loss = tot_loss_accum / max(1, n_batches)
        avg_sdr_loss = sdr_loss_accum / max(1, n_batches)
        elapsed = time.time() - t0

        raw_val, enh_val = evaluate_model(model, val_pairs, device)
        delta = enh_val - raw_val

        print(
            f"{epoch:^7d} | {avg_loss:^12.4f} | {avg_sdr_loss:^12.4f} | {enh_val:^+12.2f} | {delta:^+9.2f} | {elapsed:^7.1f}s",
            flush=True,
        )

        if enh_val > best_sdr:
            best_sdr = enh_val
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_si_sdr": best_sdr,
                    "delta_sdr": delta,
                    "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                },
                args.save_ckpt,
            )
            print(f"   --> New Best Checkpoint: {best_sdr:+.2f} dB (Saved to {args.save_ckpt})")

    print("=" * 85)
    print(f"[TRAINING COMPLETE] Peak Held-out SI-SDR: {best_sdr:+.2f} dB (Saved to {args.save_ckpt})")


if __name__ == "__main__":
    main()
