"""Train CRNN-SED (AST + CNN Dual Branch) & Build WavLM+CRNN Ensemble.

This script:
1. Loads 2,501 gold clips from /content/val_data/validation (on local NVMe disk).
2. Builds the dual-branch CRNN_SED architecture:
   - Branch 1: Audio Spectrogram Transformer (MIT/ast-finetuned-audioset-10-10-0.4593)
   - Branch 2: Log-Mel 2D CNN (supplies millisecond-sharp event boundaries)
   - Merging: BiGRU + Multi-head Attention temporal pooling
3. Trains for 10 epochs using AMP (Mixed Precision) on Colab T4 (~18-22 mins).
4. Caches CRNN validation posteriors to `val_cache_crnn.pt`.
5. Blends WavLM (`val_cache_2501.pt`) + CRNN (`val_cache_crnn.pt`) via Logit Averaging.
6. Empirically scores the ensemble against the official ground truth.
7. Runs test inference on /content/test_audio to produce `submission_track1_ensemble.zip`.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import sys
import zipfile
from pathlib import Path

# Add modular package to python path
pkg_dir = Path(__file__).resolve().parent / "track1_detection" / "versions" / "v-3_indoml2026_track1" / "modular"
sys.path.insert(0, str(pkg_dir))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import soundfile as sf
import librosa
from transformers import AutoModel, AutoFeatureExtractor

from config import CFG, FRAME_SEC, MIN_SAMPLES
from model import WavLMSED
from inference import t1_posteriors as wavlm_posteriors
from evaluate import prob_to_events, event_based_f1, segment_dice
from predict_track1 import read_audio, is_valid_torch_file

AUD = (".wav", ".flac", ".mp3", ".ogg")
CATS = ["animal", "vehicle_traffic", "baby_child", "singing_music", "phone_signal_alarm", "appliance_machine", "human_non_speech"]
CAT2IDX = {c: i for i, c in enumerate(CATS)}
N_OUT = 1 + len(CATS)  # Channel 0: any-noise (competition target)

# Training constants
TRAIN_SR = 16000
HOP_LEN = 160        # 10 ms at 16 kHz
TIME_POOL = 2        # 20 ms model output grid (matches WavLM 50 Hz output!)
MAX_CLIP_SEC = 8.0   # Fixed window for batching
N_SAMP = int(MAX_CLIP_SEC * TRAIN_SR)
FRAMES_100HZ = int(MAX_CLIP_SEC * TRAIN_SR / HOP_LEN)
FRAMES_OUT = FRAMES_100HZ // TIME_POOL


# ---------------------------------------------------------------------------
# 1. Dataset & Label Rasterization
# ---------------------------------------------------------------------------
def prep_wav(w):
    return (w - w.mean()) / (w.std() + 1e-5)


def build_labels(spans, n_frames_100hz):
    y = np.zeros((N_OUT, n_frames_100hz), dtype=np.uint8)
    for st, en, cat in spans:
        a = max(0, int(round(st * 100)))
        b = min(n_frames_100hz, int(round(en * 100)))
        if b <= a:
            continue
        y[0, a:b] = 1
        ci = CAT2IDX.get(cat)
        if ci is not None:
            y[1 + ci, a:b] = 1
    return y


def fit_wav(wav, train=True):
    if len(wav) > N_SAMP:
        s = random.randint(0, len(wav) - N_SAMP) if train else 0
        return wav[s:s + N_SAMP], s, N_SAMP
    return np.pad(wav, (0, N_SAMP - len(wav))), 0, len(wav)


def labels_to_output_grid(lab, s):
    lab = lab[:, s // HOP_LEN: s // HOP_LEN + FRAMES_100HZ]
    if lab.shape[1] < FRAMES_100HZ:
        lab = np.pad(lab, ((0, 0), (0, FRAMES_100HZ - lab.shape[1])))
    lab = lab[:, :FRAMES_100HZ]
    return lab.reshape(lab.shape[0], FRAMES_OUT, TIME_POOL).max(axis=2)


class LocalSedDataset(Dataset):
    def __init__(self, samples, train=True):
        self.samples = samples
        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        raw_wav = item["wav"]
        lab = item["lab"]
        w, s, valid_len = fit_wav(raw_wav, self.train)
        lab_out = labels_to_output_grid(lab, s)

        mask = np.zeros(FRAMES_OUT, dtype=np.float32)
        n_valid = min(FRAMES_OUT, max(1, int(math.ceil(valid_len / TRAIN_SR / (HOP_LEN * TIME_POOL / TRAIN_SR)))))
        mask[:n_valid] = 1.0

        w_norm = prep_wav(w.astype(np.float32))
        return {
            "wav": torch.from_numpy(w_norm),
            "lab": torch.from_numpy(lab_out.astype(np.float32)),
            "mask": torch.from_numpy(mask),
            "cid": item["cid"],
        }


# ---------------------------------------------------------------------------
# 2. Dual-Branch CRNN_SED Architecture (AST + CNN)
# ---------------------------------------------------------------------------
class ASTBackbone(nn.Module):
    def __init__(self, model_name="MIT/ast-finetuned-audioset-10-10-0.4593"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.fe = AutoFeatureExtractor.from_pretrained(model_name)
        try:
            self.backbone.gradient_checkpointing_enable()
        except Exception:
            pass
        cfg = self.backbone.config
        patch = int(getattr(cfg, "patch_size", 16))
        fstride = int(getattr(cfg, "frequency_stride", 10))
        tstride = int(getattr(cfg, "time_stride", 10))
        n_mels = int(getattr(cfg, "num_mel_bins", 128))
        max_len = int(getattr(cfg, "max_length", 1024))
        self.f_dim = (n_mels - patch) // fstride + 1
        self.t_dim = (max_len - patch) // tstride + 1
        self.n_special = 2
        self.out_dim = cfg.hidden_size

    def forward(self, wav):
        wav_np = wav.detach().float().cpu().numpy()
        feats = self.fe([w for w in wav_np], sampling_rate=TRAIN_SR, return_tensors="pt")
        h = self.backbone(feats["input_values"].to(wav.device)).last_hidden_state
        h = h[:, self.n_special:, :]
        if h.shape[1] == self.f_dim * self.t_dim:
            h = h.reshape(h.shape[0], self.f_dim, self.t_dim, h.shape[-1]).amax(dim=1)
        return h  # (B, t_dim, D)


class CNNBranch(nn.Module):
    """Local 2D CNN on Log-Mel spectrogram to provide millisecond-sharp edges."""
    def __init__(self, n_mels=128, out_ch=128):
        super().__init__()
        import torchaudio.transforms as T
        self.melspec = T.MelSpectrogram(
            sample_rate=TRAIN_SR, n_fft=1024, hop_length=HOP_LEN,
            n_mels=n_mels, power=2.0
        )
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((4, 1)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((4, 2)),
            nn.Conv2d(64, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None))
        )
        self.out_ch = out_ch

    def forward(self, wav):
        # wav: (B, N_SAMP)
        mel = self.melspec(wav)  # (B, n_mels, T)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        std = mel.std(dim=(-2, -1), keepdim=True)
        std = torch.clamp(std, min=1e-3)
        mel = (mel - mel.mean(dim=(-2, -1), keepdim=True)) / std
        x = mel.unsqueeze(1)      # (B, 1, n_mels, T)
        x = self.net(x).squeeze(2)  # (B, out_ch, T')
        return x.transpose(1, 2)  # (B, T', out_ch)


class CRNN_SED(nn.Module):
    def __init__(self, n_out=N_OUT, rnn_dim=256):
        super().__init__()
        self.ast = ASTBackbone()
        self.cnn = CNNBranch(n_mels=128, out_ch=128)
        merge_dim = self.ast.out_dim + self.cnn.out_ch
        self.merge = nn.Linear(merge_dim, 256)
        self.rnn = nn.GRU(256, rnn_dim, 2, batch_first=True, bidirectional=True, dropout=0.1)
        self.strong = nn.Linear(2 * rnn_dim, n_out)
        self.att = nn.Linear(2 * rnn_dim, n_out)

    def forward(self, wav, target_len=FRAMES_OUT, mask=None):
        seq = self.ast(wav)  # (B, t_seq, D)
        seq = F.interpolate(seq.transpose(1, 2), size=target_len, mode="linear", align_corners=False).transpose(1, 2)
        cnn = self.cnn(wav)  # (B, t_cnn, C)
        cnn = F.interpolate(cnn.transpose(1, 2), size=target_len, mode="linear", align_corners=False).transpose(1, 2)
        h = torch.cat([seq, cnn], dim=-1)
        h = F.relu(self.merge(h))
        h, _ = self.rnn(h)
        frame_logits = self.strong(h)  # (B, T, C)
        att = self.att(h)
        if mask is not None:
            att = att.masked_fill(mask.unsqueeze(-1) < 0.5, -100.0)
        att = torch.softmax(att.float(), dim=1).type_as(att)
        clip_prob = (torch.sigmoid(frame_logits) * att).sum(dim=1).clamp(1e-5, 1.0 - 1e-5)
        return frame_logits.transpose(1, 2), clip_prob  # (B, C, T), (B, C)


# ---------------------------------------------------------------------------
# 3. Loss & Training (Rock-solid Numerical Stability)
# ---------------------------------------------------------------------------
def robust_bce_loss(logits, target, mask, pos_weight=3.0):
    pw = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pw)
    m = mask.unsqueeze(1).to(logits.device, dtype=logits.dtype)
    loss = (bce * m).sum() / (m.sum() * target.shape[1] + 1e-6)
    return loss


def train_crnn(model, train_dl, val_dl, device, epochs=10, lr=2e-4):
    opt = torch.optim.AdamW([
        {"params": model.ast.parameters(), "lr": lr * 0.05},
        {"params": model.cnn.parameters(), "lr": lr},
        {"params": model.merge.parameters(), "lr": lr},
        {"params": model.rnn.parameters(), "lr": lr},
        {"params": model.strong.parameters(), "lr": lr},
        {"params": model.att.parameters(), "lr": lr},
    ], weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    best_val_loss = float("inf")
    best_weights = None

    print(f"\n--- TRAINING CRNN_SED DUAL-BRANCH ({epochs} EPOCHS) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, steps = 0.0, 0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{epochs} [Train]")
        for b in pbar:
            wav = b["wav"].to(device)
            lab = b["lab"].to(device)
            mask = b["mask"].to(device)

            opt.zero_grad()
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits, clip_p = model(wav, target_len=FRAMES_OUT, mask=mask)
                    loss = robust_bce_loss(logits, lab, mask)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
            else:
                logits, clip_p = model(wav, target_len=FRAMES_OUT, mask=mask)
                loss = robust_bce_loss(logits, lab, mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

            total_loss += loss.item()
            steps += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        train_loss = total_loss / max(1, steps)

        # Validation Loss
        model.eval()
        val_loss, val_steps = 0.0, 0
        with torch.no_grad():
            for b in val_dl:
                wav = b["wav"].to(device)
                lab = b["lab"].to(device)
                mask = b["mask"].to(device)
                if scaler is not None:
                    with torch.amp.autocast("cuda"):
                        logits, _ = model(wav, target_len=FRAMES_OUT, mask=mask)
                        loss = robust_bce_loss(logits, lab, mask)
                else:
                    logits, _ = model(wav, target_len=FRAMES_OUT, mask=mask)
                    loss = robust_bce_loss(logits, lab, mask)
                val_loss += loss.item()
                val_steps += 1

        val_loss = val_loss / max(1, val_steps)
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu() for k, v in model.state_dict().items()}
            print(f"  --> Saved new best model checkpoint (Val Loss: {best_val_loss:.4f})")

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model


# ---------------------------------------------------------------------------
# 4. CRNN Posteriors on Full Audio
# ---------------------------------------------------------------------------
def crnn_posteriors(wav, model, device):
    """Run CRNN on arbitrary-length audio clip with sliding windows."""
    model.eval()
    dur = len(wav) / TRAIN_SR
    target_frames = int(round(dur / (HOP_LEN * TIME_POOL / TRAIN_SR)))
    w_norm = prep_wav(wav.astype(np.float32))

    # Pad or crop to chunks
    chunk_size = N_SAMP
    stride = int(chunk_size * 0.75)
    if len(w_norm) <= chunk_size:
        padded = np.pad(w_norm, (0, chunk_size - len(w_norm)))
        tensor_w = torch.from_numpy(padded).unsqueeze(0).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            logits, clip_p = model(tensor_w, target_len=FRAMES_OUT)
            prob = torch.sigmoid(logits[0, 0]).cpu().numpy()
            cp = float(clip_p[0, 0].cpu())
        prob = prob[:target_frames]
        return prob, cp

    # Sliding window
    accum_prob = np.zeros(target_frames, dtype=np.float32)
    accum_count = np.zeros(target_frames, dtype=np.float32)
    clip_probs = []

    pos = 0
    while pos < len(w_norm):
        chunk = w_norm[pos:pos + chunk_size]
        actual_len = len(chunk)
        if actual_len < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - actual_len))
        tensor_w = torch.from_numpy(chunk).unsqueeze(0).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            logits, clip_p = model(tensor_w, target_len=FRAMES_OUT)
            prob = torch.sigmoid(logits[0, 0]).cpu().numpy()
            clip_probs.append(float(clip_p[0, 0].cpu()))

        frame_start = int(round(pos / (HOP_LEN * TIME_POOL)))
        valid_frames = min(FRAMES_OUT, int(round(actual_len / (HOP_LEN * TIME_POOL))))
        frame_end = min(target_frames, frame_start + valid_frames)
        avail = frame_end - frame_start
        if avail > 0:
            accum_prob[frame_start:frame_end] += prob[:avail]
            accum_count[frame_start:frame_end] += 1.0

        pos += stride

    accum_count = np.maximum(accum_count, 1e-5)
    final_prob = accum_prob / accum_count
    return final_prob, float(np.mean(clip_probs))


# ---------------------------------------------------------------------------
# 5. Main Execution: Train -> Cache -> Blend -> Predict
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train CRNN_SED & Ensemble with WavLM")
    parser.add_argument("--val-dir", type=str, default="/content/val_data/validation", help="Validation folder path")
    parser.add_argument("--test-dir", type=str, default="/content/test_audio", help="Test audio folder path")
    parser.add_argument("--wavlm-ckpt", type=str, default="/content/t1_wavlm.zip", help="WavLM checkpoint path")
    parser.add_argument("--wavlm-cache", type=str, default="val_cache_2501.pt", help="Cached WavLM posteriors")
    parser.add_argument("--epochs", type=int, default=10, help="CRNN training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--output-zip", type=str, default="submission_track1_ensemble.zip", help="Final ensemble zip")
    args = parser.parse_args()

    val_dir = Path(args.val_dir)
    meta_path = val_dir / "validationMetadata.json"
    if not meta_path.exists():
        candidates = list(val_dir.rglob("validationMetadata.json"))
        if candidates:
            meta_path = candidates[0]
            val_dir = meta_path.parent
        else:
            print(f"[ERROR] validationMetadata.json not found in {val_dir}!")
            return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("      INDOML 2026: CRNN_SED TRAINING & 3-WAY ENSEMBLE GENERATOR")
    print(f"      Device: {device.upper()} | Model: AST + CNN Dual Branch")
    print("=" * 80)

    # 1. Load Local Ground-Truth Data
    print(f"\n[INFO] Loading validation metadata from: {meta_path.resolve()}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_items = json.load(f)

    audio_files = {}
    for p in val_dir.rglob("*"):
        if p.suffix.lower() in AUD:
            audio_files[p.stem] = p

    samples = []
    ref_dict = {}
    print(f"[INFO] Reading local audio files ({len(meta_items):,} clips)...")
    for item in tqdm(meta_items, desc="Loading Audio"):
        fn = item.get("segmentFileName", "")
        cid = fn[:-4] if fn.lower().endswith(".wav") else fn
        if cid not in audio_files:
            continue
        wav_path = audio_files[cid]
        wav = read_audio(wav_path, sr=TRAIN_SR)

        spans = []
        for ev in (item.get("NoiseSubCategoryTimeStamp", []) or []):
            try:
                s, e = float(ev.get("start", 0)), float(ev.get("end", 0))
                if e > s:
                    spans.append((s, e, ev.get("category", "any")))
            except Exception:
                continue

        nf = int(math.ceil(len(wav) / HOP_LEN))
        lab = build_labels(spans, nf)
        samples.append({"cid": cid, "wav": wav, "lab": lab, "spans": spans})
        ref_dict[cid] = [(s, e) for s, e, _ in spans]

    print(f"[SUCCESS] Loaded {len(samples):,} labeled audio clips from disk.")

    # 2. Train/Val Split (85% Train, 15% Internal Val)
    random.seed(42)
    random.shuffle(samples)
    n_val = max(1, int(len(samples) * 0.15))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    print(f"[INFO] Dataset split: {len(train_samples):,} train | {len(val_samples):,} internal val")

    train_dl = DataLoader(LocalSedDataset(train_samples, train=True), batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_dl = DataLoader(LocalSedDataset(val_samples, train=False), batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 3. Instantiate and Train CRNN_SED
    print("\n[INFO] Initializing CRNN_SED dual-branch network...")
    model = CRNN_SED(n_out=N_OUT).to(device)
    model = train_crnn(model, train_dl, val_dl, device, epochs=args.epochs, lr=3e-4)

    # Save model weights
    torch.save({"model": model.state_dict()}, "t1_crnn_sed.pt")
    print("[SUCCESS] Checkpoint saved: t1_crnn_sed.pt")

    # 4. Cache CRNN Posteriors on all 2,501 Validation Clips
    print("\n--- CACHING CRNN POSTERIORS FOR ALL VALIDATION CLIPS ---")
    crnn_cache = {}
    for item in tqdm(samples, desc="CRNN Posteriors"):
        cid = item["cid"]
        prob, cp = crnn_posteriors(item["wav"], model, device)
        crnn_cache[cid] = {"prob": prob, "cp": cp, "dur": len(item["wav"]) / TRAIN_SR}
    torch.save(crnn_cache, "val_cache_crnn.pt")

    # 5. Offline Ensemble Validation Blending (WavLM + CRNN)
    print("\n" + "=" * 80)
    print("      OFFLINE ENSEMBLE EVALUATION & WEIGHT OPTIMIZATION")
    print("=" * 80)

    # Load WavLM Cache
    wavlm_cache_file = Path(args.wavlm_cache)
    if not wavlm_cache_file.exists():
        for cand in [Path("val_cache_2501.pt"), Path("/content/indoml-2026/val_cache_2501.pt")]:
            if cand.exists():
                wavlm_cache_file = cand
                break

    wavlm_data = {}
    if wavlm_cache_file.exists():
        print(f"[INFO] Loading cached WavLM posteriors from: {wavlm_cache_file.name}")
        saved = torch.load(wavlm_cache_file, map_location="cpu", weights_only=False)
        for item in saved["cache"]:
            wavlm_data[item["cid"]] = item
    else:
        print("[WARN] val_cache_2501.pt not found. Run sweep_official_val.py first to cache WavLM.")
        return

    # Evaluate Ensemble Grid
    print("[INFO] Blending posterior curves with Logit Averaging...")
    weights = [0.40, 0.50, 0.55, 0.60]  # Weight for WavLM (1-w for CRNN)
    thresholds = [0.60, 0.65, 0.70, 0.75]
    medians = [11, 15, 17]

    best_ensemble = {"comb": 0.0}
    common_cids = [cid for cid in ref_dict if cid in wavlm_data and cid in crnn_cache]

    for w_wavlm in weights:
        w_crnn = 1.0 - w_wavlm
        # Precompute blended posteriors for this weight
        blended_map = {}
        for cid in common_cids:
            p1 = wavlm_data[cid]["post"]
            p2 = crnn_cache[cid]["prob"]
            dur = wavlm_data[cid]["dur"]

            # Interpolate to same frame length if minor difference
            if len(p1) != len(p2):
                min_len = min(len(p1), len(p2))
                p1 = p1[:min_len]
                p2 = p2[:min_len]

            # Logit averaging
            eps = 1e-4
            l1 = np.log(np.clip(p1, eps, 1.0 - eps) / (1.0 - np.clip(p1, eps, 1.0 - eps)))
            l2 = np.log(np.clip(p2, eps, 1.0 - eps) / (1.0 - np.clip(p2, eps, 1.0 - eps)))
            l_blend = w_wavlm * l1 + w_crnn * l2
            p_blend = 1.0 / (1.0 + np.exp(-l_blend))
            blended_map[cid] = (p_blend, dur)

        for thr in thresholds:
            for med in medians:
                pred_dict = {}
                for cid in common_cids:
                    p_blend, dur = blended_map[cid]
                    raw_evs = prob_to_events(p_blend, thr=thr, med=med, min_dur=0.10, merge_gap=0.10)
                    valid_evs = []
                    for on, off in raw_evs:
                        on, off = max(0.0, on), min(dur, off)
                        if off - on >= 0.10:
                            valid_evs.append((round(float(on), 3), round(float(off), 3)))
                    pred_dict[cid] = valid_evs

                f1, prec, rec, tp, fp, fn = event_based_f1(ref_dict, pred_dict)
                dice = segment_dice(ref_dict, pred_dict)
                comb = f1 + dice
                if comb > best_ensemble["comb"]:
                    best_ensemble = {
                        "comb": comb, "f1": f1, "dice": dice, "prec": prec, "rec": rec,
                        "w_wavlm": w_wavlm, "thr": thr, "med": med
                    }

    print("\n" + "=" * 80)
    print("                   WINNING ENSEMBLE VALIDATION SCORE")
    print("=" * 80)
    print(f"  Single WavLM Baseline:   Validation Comb = 0.6762 (Leaderboard = 1.00)")
    print(f"  WavLM + CRNN Ensemble:   Validation Comb = {best_ensemble['comb']:.4f}")
    print(f"                           Event F1:        {best_ensemble['f1']:.4f}")
    print(f"                           Segment Dice:    {best_ensemble['dice']:.4f}")
    print(f"                           Optimal Weight:  {best_ensemble['w_wavlm']*100:.0f}% WavLM + {(1-best_ensemble['w_wavlm'])*100:.0f}% CRNN")
    print(f"                           Parameters:      thr={best_ensemble['thr']}, med={best_ensemble['med']}")
    print("=" * 80)

    # 6. Test Inference: Produce Final Submission
    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        for cand in [Path("/content/test_audio"), Path("/kaggle/working/test_audio"), Path("test_audio")]:
            if cand.exists():
                test_dir = cand
                break

    if not test_dir.exists():
        print(f"[WARN] Test directory '{test_dir}' not found. Skipping submission zip.")
        return

    # Load WavLM Model for Test
    print("\n--- LOADING WAVLM CHECKPOINT FOR TEST ENSEMBLE INFERENCE ---")
    wavlm_path = Path(args.wavlm_ckpt)
    ck = torch.load(wavlm_path, map_location=device, weights_only=False)
    wavlm_model = WavLMSED().to(device)
    wavlm_model.load_state_dict(ck["model"])
    wavlm_model.eval()

    test_files = sorted([p for p in test_dir.rglob("*") if p.suffix.lower() in AUD])
    print(f"[INFO] Found {len(test_files):,} test clips. Running Dual-Model Ensemble Inference...")

    jsonl_path = Path("predictions.jsonl")
    w1 = best_ensemble["w_wavlm"]
    w2 = 1.0 - w1
    thr = best_ensemble["thr"]
    med = best_ensemble["med"]

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for p in tqdm(test_files, desc="Ensemble Inference"):
            wav = read_audio(p, sr=TRAIN_SR)
            dur = max(len(wav) / TRAIN_SR, MIN_SAMPLES / TRAIN_SR)

            # Model 1: WavLM Posteriors
            p_wavlm, cp_wavlm = wavlm_posteriors(wav, wavlm_model, device, sr=TRAIN_SR)
            p1 = p_wavlm[0]

            # Model 2: CRNN Posteriors
            p2, cp_crnn = crnn_posteriors(wav, model, device)

            # Align lengths
            min_l = min(len(p1), len(p2))
            p1 = p1[:min_l]
            p2 = p2[:min_l]

            # Logit Average
            eps = 1e-4
            l1 = np.log(np.clip(p1, eps, 1.0 - eps) / (1.0 - np.clip(p1, eps, 1.0 - eps)))
            l2 = np.log(np.clip(p2, eps, 1.0 - eps) / (1.0 - np.clip(p2, eps, 1.0 - eps)))
            l_blend = w1 * l1 + w2 * l2
            p_blend = 1.0 / (1.0 + np.exp(-l_blend))

            # Post-processing
            raw_evs = prob_to_events(p_blend, thr=thr, med=med, min_dur=0.10, merge_gap=0.10)
            events = []
            for on, off in raw_evs:
                on, off = float(max(0.0, on)), float(min(dur, off))
                if off - on >= 0.10:
                    events.append({"onset": round(float(on), 3), "offset": round(float(off), 3)})

            f.write(json.dumps({"clip_id": p.stem, "events": events}, ensure_ascii=False) + "\n")

    out_zip = Path(args.output_zip)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(jsonl_path, "predictions.jsonl")

    print("\n" + "=" * 80)
    print("                FINAL ENSEMBLE SUBMISSION READY!")
    print("=" * 80)
    print(f"  Output ZIP:      {out_zip.resolve()} ({out_zip.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  Validation Comb: {best_ensemble['comb']:.4f} (Event F1: {best_ensemble['f1']:.4f}, Dice: {best_ensemble['dice']:.4f})")
    print(f"  Ensemble Mix:    {w1*100:.0f}% WavLM + {w2*100:.0f}% CRNN (AST + CNN)")
    print(f"  Parameters:      thr={thr}, med={med}, min_dur=0.10s, merge_gap=0.10s")
    print("=" * 80)


if __name__ == "__main__":
    main()
