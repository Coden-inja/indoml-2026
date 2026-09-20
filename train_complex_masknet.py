"""Complex Ratio MaskNet (cRM) with HF Simulated Pretraining & Real Fine-Tuning.

Two-Stage Championship Training for IndoML 2026 Track 2:
  Stage 1 (Pre-training): Streams Gold-tier audio from ARTPARK-IISc/Vaani-Noise-Event-Dataset,
                          extracts clean speech spans and noise event spans, and trains on
                          thousands of dynamically simulated mixtures across random SNRs.
  Stage 2 (Fine-tuning):  Fine-tunes directly on the 695 official organizer pairs with
                          active phase cancellation (Complex Ratio Masking) and Multi-Res STFT loss.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Signal Processing Parameters
# ---------------------------------------------------------------------------
SR = 16000
N_FFT = 512
HOP_LEN = 160
WIN_LEN = 512
CHUNK_SAMP = 64000  # 4-second slices during training
REPO = "ARTPARK-IISc/Vaani-Noise-Event-Dataset"


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Losses & Metrics
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

            sc_loss = torch.norm(ref_mag - est_mag, p="fro") / (torch.norm(ref_mag, p="fro") + 1e-7)
            log_loss = F.l1_loss(torch.log(est_mag), torch.log(ref_mag))
            total_loss += sc_loss + log_loss
        return total_loss / len(self.fft_sizes)


def calc_si_sdr_np(ref: np.ndarray, est: np.ndarray, eps: float = 1e-8) -> float:
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
        B, T = x.shape
        stft_c = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            win_length=self.win_len,
            window=self.window,
            return_complex=True,
        )
        xr = stft_c.real
        xi = stft_c.imag
        xmag = torch.abs(stft_c)

        feat = torch.cat([xr, xi, xmag], dim=1).permute(0, 2, 1)
        h = self.in_proj(feat)
        gru_out, _ = self.gru(h)

        mr = self.real_head(gru_out).permute(0, 2, 1) * 2.0
        mi = self.imag_head(gru_out).permute(0, 2, 1) * 2.0

        sr = mr * xr - mi * xi
        si = mr * xi + mi * xr
        s_c = torch.complex(sr, si)

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
# HF Bank Builder & Dynamic Mixture Generator
# ---------------------------------------------------------------------------
def decode_hf_audio(cell, target_sr=16000):
    if isinstance(cell, dict):
        if cell.get("bytes"):
            w, sr = sf.read(io.BytesIO(cell["bytes"]), dtype="float32")
        elif cell.get("path"):
            w, sr = sf.read(cell["path"], dtype="float32")
        elif cell.get("array") is not None:
            w, sr = np.asarray(cell["array"], dtype=np.float32), cell["sampling_rate"]
        else:
            return None
    else:
        return None
    if w.ndim > 1:
        w = w.mean(axis=1)
    if sr != target_sr:
        import librosa
        w = librosa.resample(w, orig_sr=sr, target_sr=target_sr)
    return w.astype(np.float32)


def build_hf_banks(max_gold=1200, bank_dir="/kaggle/working/banks", token=None):
    from datasets import load_dataset, Audio

    clean_dir = Path(bank_dir) / "clean"
    noise_dir = Path(bank_dir) / "noise"
    clean_dir.mkdir(parents=True, exist_ok=True)
    noise_dir.mkdir(parents=True, exist_ok=True)

    clean_idx_path = Path(bank_dir) / "clean_index.json"
    noise_idx_path = Path(bank_dir) / "noise_index.json"

    if clean_idx_path.exists() and noise_idx_path.exists():
        with open(clean_idx_path) as f:
            c_idx = json.load(f)
        with open(noise_idx_path) as f:
            n_idx = json.load(f)
        if len(c_idx) > 200 and len(n_idx) > 200:
            print(f"[INFO] Loaded existing audio banks: {len(c_idx)} clean, {len(n_idx)} noise.")
            return c_idx, n_idx

    print(f"[INFO] Streaming {max_gold} Gold clips from HF to build clean & noise banks...")
    raw = load_dataset(REPO, split="train", streaming=True, token=token)
    raw = raw.cast_column("audio", Audio(decode=False))

    clean_idx, noise_idx, n_gold = [], [], 0
    t0 = time.time()

    for i, ex in enumerate(raw):
        if ex.get("annotationQuality") != "verified_timestamps":
            continue
        if n_gold >= max_gold:
            break

        try:
            wav = decode_hf_audio(ex["audio"])
        except Exception:
            continue
        if wav is None or len(wav) < SR * 1.0:
            continue

        n_gold += 1
        dur = len(wav) / SR
        stamps = ex.get("NoiseSubCategoryTimeStamp") or []
        spans = []
        for s in stamps:
            try:
                st, en = float(s["start"]), float(s["end"])
                if en > st:
                    spans.append((st, en))
            except Exception:
                continue
        spans = sorted(spans)

        # 1. Clean spans: gaps between events
        cur = 0.0
        for k, (st, en) in enumerate(spans):
            if st - cur >= 1.5:
                seg = wav[int(cur * SR) : int(st * SR)]
                if np.abs(seg).max() > 1e-3:
                    p = clean_dir / f"c_{n_gold:05d}_{k}.wav"
                    sf.write(p, seg, SR)
                    clean_idx.append(str(p))
            cur = max(cur, en)
        if dur - cur >= 1.5:
            seg = wav[int(cur * SR) : int(dur * SR)]
            if np.abs(seg).max() > 1e-3:
                p = clean_dir / f"c_{n_gold:05d}_end.wav"
                sf.write(p, seg, SR)
                clean_idx.append(str(p))

        # 2. Noise spans: the events
        for k, (st, en) in enumerate(spans):
            if 0.3 <= en - st <= 5.0:
                seg = wav[int(st * SR) : int(en * SR)]
                if np.abs(seg).max() > 1e-3:
                    p = noise_dir / f"n_{n_gold:05d}_{k}.wav"
                    sf.write(p, seg, SR)
                    noise_idx.append(str(p))

        if n_gold % 200 == 0:
            print(f"  [HF Bank] Processed {n_gold}/{max_gold} clips ({len(clean_idx)} clean, {len(noise_idx)} noise)...")

    with open(clean_idx_path, "w") as f:
        json.dump(clean_idx, f)
    with open(noise_idx_path, "w") as f:
        json.dump(noise_idx, f)

    print(f"[INFO] Built banks in {(time.time()-t0)/60:.1f} mins: {len(clean_idx)} clean, {len(noise_idx)} noise.")
    return clean_idx, noise_idx


class DynamicMixtureDataset(Dataset):
    def __init__(self, clean_paths, noise_paths, length=6000, chunk_samp=CHUNK_SAMP):
        self.clean_paths = clean_paths
        self.noise_paths = noise_paths
        self.length = length
        self.chunk_samp = chunk_samp

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        cp = random.choice(self.clean_paths)
        clean, _ = sf.read(cp, dtype="float32")
        if len(clean) < self.chunk_samp:
            clean = np.pad(clean, (0, self.chunk_samp - len(clean)))
        else:
            s = random.randint(0, len(clean) - self.chunk_samp)
            clean = clean[s : s + self.chunk_samp]

        # Sample noise and mix at random SNR [-6 dB, +12 dB]
        np_ = random.choice(self.noise_paths)
        noise, _ = sf.read(np_, dtype="float32")
        if len(noise) < self.chunk_samp:
            noise = np.tile(noise, int(np.ceil(self.chunk_samp / max(1, len(noise)))))[: self.chunk_samp]
        else:
            s = random.randint(0, len(noise) - self.chunk_samp)
            noise = noise[s : s + self.chunk_samp]

        clean_pwr = np.mean(clean**2) + 1e-8
        noise_pwr = np.mean(noise**2) + 1e-8
        target_snr_db = random.uniform(-6.0, 12.0)
        target_noise_pwr = clean_pwr / (10.0 ** (target_snr_db / 10.0))
        scale = np.sqrt(target_noise_pwr / noise_pwr)
        
        mix = clean + scale * noise
        pk = np.abs(mix).max()
        if pk > 0.99:
            mix = mix / pk * 0.95
            clean = clean / pk * 0.95

        return torch.from_numpy(mix.astype(np.float32)), torch.from_numpy(clean.astype(np.float32))


class RealAudioPairDataset(Dataset):
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
            if random.random() < 0.35:
                noise_part = noisy - clean
                noisy = clean + random.uniform(0.5, 1.5) * noise_part
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
# Evaluation
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
        est, _, _ = model(inp)
        enh = est[0].cpu().numpy()

        raw_sdrs.append(calc_si_sdr_np(clean, noisy))
        enh_sdrs.append(calc_si_sdr_np(clean, enh))
    return float(np.mean(raw_sdrs)), float(np.mean(enh_sdrs))


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Complex Ratio MaskNet (cRM) Two-Stage Pipeline")
    parser.add_argument("--val-root", type=str, default="/kaggle/working/val_data/validation")
    parser.add_argument("--hf-pretrain", action="store_true", help="Enable Stage 1 pretraining on HF dynamic mixtures")
    parser.add_argument("--max-gold", type=int, default=1000, help="Gold clips to stream for bank generation")
    parser.add_argument("--sim-samples", type=int, default=5000, help="Number of simulated mixtures to generate")
    parser.add_argument("--pretrain-epochs", type=int, default=25, help="Stage 1 pretrain epochs")
    parser.add_argument("--finetune-epochs", type=int, default=60, help="Stage 2 fine-tune epochs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--stft-weight", type=float, default=0.20)
    parser.add_argument("--save-ckpt", type=str, default="/kaggle/working/t2_complex_masknet_best.pt")
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using compute device: {device}")

    # Load Real Pairs
    val_root = Path(args.val_root)
    meta_path = val_root / "validationMetadata.json"
    with open(meta_path) as f:
        meta = json.load(f)

    synth_items = [m for m in meta if m.get("syntheticData") is True]
    clean_dir = val_root / "syntheticCleanRefAudio"
    noise_dir = val_root / "syntheticNoiseAudio"

    real_pairs = []
    for item in synth_items:
        fn = item["segmentFileName"]
        cp = clean_dir / fn
        np_ = noise_dir / fn
        if cp.exists() and np_.exists():
            real_pairs.append((str(np_), str(cp)))

    random.Random(42).shuffle(real_pairs)
    train_pairs = real_pairs[:550]
    val_pairs = real_pairs[550:]
    print(f"[INFO] Real Pairs: {len(train_pairs)} train | {len(val_pairs)} held-out validation")

    # Initialize Model
    model = ComplexMaskNet(hidden_size=args.hidden_size, num_layers=args.num_layers).to(device)
    mr_stft_loss_fn = MultiResolutionSTFTLoss().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Initialized ComplexMaskNet ({total_params / 1e6:.2f}M trainable parameters).")

    # =========================================================================
    # STAGE 1: Pre-training on HF Simulated Mixtures (Optional / Flagged)
    # =========================================================================
    if args.hf_pretrain:
        print("\n" + "=" * 80)
        print("          STAGE 1: PRE-TRAINING ON HF DYNAMIC MIXTURES")
        print("=" * 80)
        token = os.environ.get("HF_TOKEN")
        clean_banks, noise_banks = build_hf_banks(max_gold=args.max_gold, token=token)
        sim_ds = DynamicMixtureDataset(clean_banks, noise_banks, length=args.sim_samples)
        sim_loader = DataLoader(sim_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

        opt1 = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=args.pretrain_epochs * len(sim_loader))

        for ep in range(1, args.pretrain_epochs + 1):
            t0 = time.time()
            model.train()
            tot_loss = 0.0
            n_b = 0
            for mix_b, clean_b in sim_loader:
                mix_b, clean_b = mix_b.to(device), clean_b.to(device)
                opt1.zero_grad()
                est_b, _, _ = model(mix_b)
                l = si_sdr_loss_torch(est_b, clean_b) + args.stft_weight * mr_stft_loss_fn(est_b, clean_b)
                l.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt1.step()
                sched1.step()
                tot_loss += float(l.item())
                n_b += 1
            print(f"Pretrain Epoch {ep:02d}/{args.pretrain_epochs} [{time.time()-t0:.1f}s] Loss: {tot_loss/max(1,n_b):.4f}")

    # =========================================================================
    # STAGE 2: Fine-Tuning on Official Pairs
    # =========================================================================
    print("\n" + "=" * 85)
    print("          STAGE 2: FINE-TUNING ON OFFICIAL VALIDATION PAIRS")
    print("=" * 85)

    real_ds = RealAudioPairDataset(train_pairs, is_train=True)
    real_loader = DataLoader(real_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    ft_lr = args.lr if not args.hf_pretrain else args.lr * 0.5
    optimizer = torch.optim.AdamW(model.parameters(), lr=ft_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs * len(real_loader), eta_min=1e-6)

    raw_val0, enh_val0 = evaluate_model(model, val_pairs, device)
    print(f"\n[BASELINE] Held-out Raw SI-SDR: {raw_val0:+.2f} dB | Current Model: {enh_val0:+.2f} dB\n")

    best_sdr = -1e9
    print("=" * 85)
    print(f"{'Epoch':^7} | {'Total Loss':^12} | {'SI-SDR Loss':^12} | {'Val SI-SDR':^12} | {'Delta':^9} | {'Time':^8}")
    print("=" * 85)

    for epoch in range(1, args.finetune_epochs + 1):
        t0 = time.time()
        model.train()
        tot_loss_accum = 0.0
        sdr_loss_accum = 0.0
        n_batches = 0

        for noisy_b, clean_b in real_loader:
            noisy_b, clean_b = noisy_b.to(device), clean_b.to(device)

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
    print(f"[COMPLETE] Peak Held-out SI-SDR: {best_sdr:+.2f} dB (Saved to {args.save_ckpt})")


if __name__ == "__main__":
    main()
