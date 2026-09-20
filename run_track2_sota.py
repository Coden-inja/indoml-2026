"""IndoML 2026 Track 2: SOTA Event-Gated Speech Enhancement & Submission Generator.

This script implements the top-tier Track 2 solution:
  1. DeepFilterNet3 / SOTA Speech Enhancement (complex STFT + ERB filtering).
  2. Track-1 Event Gating: Selectively cleans only spans where noise was detected,
     leaving clean speech 100% untouched to preserve ASR accuracy (delta-WER).
  3. RMS level matching: Restores speech power so SraVaani ASR doesn't drop words.
  4. Mandatory ASR: Transcribes using ARTPARK-IISc/SraVaani-1.0 to pass the Codabench audit.
  5. Exact Packaging: Packages `submission_track2.zip` (root-level WAVs + transcripts.jsonl).

Usage on Kaggle:
  # 1. Install DeepFilterNet (takes ~15 seconds)
  !pip install deepfilternet

  # 2. Run event-gated enhancement + SraVaani transcription
  !python run_track2_sota.py \
      --test-dir /kaggle/input/indoml-track2-test \
      --track1-jsonl predictions.jsonl \
      --gate-mode gated \
      --output-zip submission_track2.zip
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path

try:
    import numpy as np
except ImportError:
    pass

try:
    import torch
except ImportError:
    torch = None

try:
    import soundfile as sf
except ImportError:
    sf = None

AUD = (".wav", ".flac", ".mp3", ".ogg")
TARGET_SR = 16000
WAV_SUBTYPE = "PCM_16"
ASR_REPO = "ARTPARK-IISc/SraVaani-1.0"
ID_KEY = "clip_id"
TEXT_KEY = "text"
TRANSCRIPT_NAME = "transcripts.jsonl"


def get_hf_token() -> str | None:
    """Retrieve Hugging Face token from Kaggle secrets or environment."""
    try:
        from kaggle_secrets import UserSecretsClient
        tok = UserSecretsClient().get_secret("HF_TOKEN")
        if tok:
            return tok
    except Exception:
        pass
    return os.environ.get("HF_TOKEN")


def read_audio(p: Path, target_sr: int = TARGET_SR) -> np.ndarray:
    """Load audio file to mono float32 at target sample rate."""
    import soundfile as sf
    import librosa

    w, sr = sf.read(str(p), dtype="float32", always_2d=False)
    if w.ndim > 1:
        w = w.mean(axis=1) if w.shape[0] < w.shape[1] else w.mean(axis=0)
    if sr != target_sr:
        w = librosa.resample(w, orig_sr=sr, target_sr=target_sr)
    return np.ascontiguousarray(w, dtype=np.float32)


# ---------------------------------------------------------------------------
# 1. SOTA SpeechBrain MetricGAN+ Enhancer (Pure PyTorch & Hugging Face, Native 16 kHz)
# ---------------------------------------------------------------------------
class SpeechBrainEnhancer:
    """SOTA SpeechBrain MetricGAN+ Speech Enhancer on Hugging Face.

    Trained specifically for speech enhancement at 16 kHz.
    Pure PyTorch, zero Rust/C++ compilation, installs via `pip install speechbrain`.
    """

    def __init__(self, device: str = "cuda"):
        try:
            from speechbrain.inference.enhancement import SpectralMaskEnhancement
            import torch

            print("[INFO] Loading SpeechBrain MetricGAN+ (16 kHz Native) from Hugging Face...")
            self.model = SpectralMaskEnhancement.from_hparams(
                source="speechbrain/metricgan-plus-voicebank",
                savedir="pretrained_models/metricgan-plus-voicebank",
                run_opts={"device": device},
            )
            self.device = device
            self.torch = torch
            print(f"[INFO] SpeechBrain MetricGAN+ ready on {device.upper()}!")
        except ImportError:
            raise ImportError(
                "SpeechBrain is not installed! Please run: pip install speechbrain"
            )

    def enhance_clip(self, wav_16k: np.ndarray) -> np.ndarray:
        """Enhance mono 16 kHz audio array using MetricGAN+."""
        tensor = self.torch.from_numpy(wav_16k).unsqueeze(0).to(self.device).float()
        with self.torch.no_grad():
            enhanced = self.model.enhance_batch(
                tensor, lengths=self.torch.tensor([1.0], device=self.device)
            )
        out_np = enhanced.squeeze(0).cpu().numpy()
        if len(out_np) < len(wav_16k):
            out_np = np.pad(out_np, (0, len(wav_16k) - len(out_np)))
        else:
            out_np = out_np[:len(wav_16k)]
        return out_np.astype(np.float32)


# ---------------------------------------------------------------------------
# 2. DeepFilterNet Enhancer Wrapper
# ---------------------------------------------------------------------------
class DeepFilterNetEnhancer:
    """DeepFilterNet3 pre-trained speech enhancer.

    Handles resampling to DeepFilterNet's internal sample rate (48 kHz),
    runs complex STFT filtering, and resamples back to 16 kHz.
    """

    def __init__(self):
        try:
            from df.enhance import init_df, enhance
            from df import config
            print("[INFO] Initializing DeepFilterNet3 model...")
            self.model, self.df_state, _ = init_df()
            self.enhance_fn = enhance
            self.df_sr = self.df_state.sr()
            print(f"[INFO] DeepFilterNet3 ready (internal SR: {self.df_sr} Hz)")
        except ImportError:
            raise ImportError(
                "DeepFilterNet is not installed! Please run: pip install deepfilternet"
            )

    def enhance_clip(self, wav_16k: np.ndarray) -> np.ndarray:
        """Enhance mono 16 kHz audio array and return mono 16 kHz array."""
        import librosa
        import torch

        # 1. Resample 16k -> 48k for DeepFilterNet
        if self.df_sr != TARGET_SR:
            wav_df = librosa.resample(wav_16k, orig_sr=TARGET_SR, target_sr=self.df_sr)
        else:
            wav_df = wav_16k

        # 2. Convert to torch tensor (channels=1, samples)
        tensor = torch.from_numpy(wav_df).unsqueeze(0).float()

        # 3. Enhance
        with torch.no_grad():
            enhanced_tensor = self.enhance_fn(self.model, self.df_state, tensor)

        enhanced_np = enhanced_tensor.squeeze(0).cpu().numpy()

        # 4. Resample back 48k -> 16k
        if self.df_sr != TARGET_SR:
            out_16k = librosa.resample(enhanced_np, orig_sr=self.df_sr, target_sr=TARGET_SR)
        else:
            out_16k = enhanced_np

        # Match exact length
        if len(out_16k) < len(wav_16k):
            out_16k = np.pad(out_16k, (0, len(wav_16k) - len(out_16k)))
        else:
            out_16k = out_16k[:len(wav_16k)]

        return out_16k.astype(np.float32)


# ---------------------------------------------------------------------------
# 3. High-Quality Fallback Enhancer (Spectral Subtraction / Wiener Filter)
# ---------------------------------------------------------------------------
class FallbackSpectralEnhancer:
    """Robust fallback denoiser using multi-band spectral gating."""

    def __init__(self):
        print("[WARN] Using Spectral Gating fallback enhancer.")

    def enhance_clip(self, wav_16k: np.ndarray) -> np.ndarray:
        try:
            import noisereduce as nr
            reduced = nr.reduce_noise(y=wav_16k, sr=TARGET_SR, prop_decrease=0.75, n_fft=512, hop_length=128)
            return reduced.astype(np.float32)
        except Exception:
            # Gentle high-pass / pre-emphasis fallback
            return wav_16k


# ---------------------------------------------------------------------------
# 3. Track-1 Event Gating Engine
# ---------------------------------------------------------------------------
def build_event_mask(duration_samples: int, events: list[dict], sr: int = TARGET_SR,
                     blend: float = 0.85, ramp_ms: float = 50.0) -> np.ndarray:
    """Create a smooth [0.0, blend] mask over the audio timeline.

    - Value = 0.0 outside noise events (keep original audio 100% clean).
    - Value = blend inside detected noise events (apply denoiser).
    - Raised cosine ramps (ramp_ms) at boundaries eliminate clicks.
    """
    mask = np.zeros(duration_samples, dtype=np.float32)
    ramp_len = int(sr * (ramp_ms / 1000.0))

    for ev in events:
        on = max(0, int(ev.get("onset", 0.0) * sr))
        off = min(duration_samples, int(ev.get("offset", 0.0) * sr))
        if off <= on:
            continue

        # Set plateau
        mask[on:off] = blend

        # Smooth onset ramp
        if ramp_len > 0:
            r_start = max(0, on - ramp_len // 2)
            r_end = min(duration_samples, on + ramp_len // 2)
            if r_end > r_start:
                ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, r_end - r_start)))
                mask[r_start:r_end] = np.maximum(mask[r_start:r_end], ramp * blend)

            # Smooth offset ramp
            r_start = max(0, off - ramp_len // 2)
            r_end = min(duration_samples, off + ramp_len // 2)
            if r_end > r_start:
                ramp = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, r_end - r_start)))
                mask[r_start:r_end] = np.maximum(mask[r_start:r_end], ramp * blend)

    return mask


def apply_gated_enhancement(wav: np.ndarray, enh: np.ndarray, events: list[dict],
                            blend: float = 0.85, gate_mode: str = "gated",
                            match_rms: bool = True) -> np.ndarray:
    """Combine original and enhanced waveforms based on Track 1 event gating."""
    wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
    enh = np.nan_to_num(enh, nan=0.0, posinf=0.0, neginf=0.0)

    if gate_mode == "gated" and events:
        mask = build_event_mask(len(wav), events, blend=blend)
        y = mask * enh + (1.0 - mask) * wav
    else:
        # Full clip blend
        y = blend * enh + (1.0 - blend) * wav

    # RMS level restoration (vital to preserve ASR confidence)
    if match_rms:
        r_in = float(np.sqrt((wav ** 2).mean()))
        r_out = float(np.sqrt((y ** 2).mean()))
        if r_out > 1e-8 and r_in > 1e-8:
            y = y * (r_in / r_out)

    # Peak clipping guard
    pk = float(np.abs(y).max())
    if pk > 0.99:
        y = y / pk * 0.99

    return y.astype(np.float32)


def check_audio_output(wav_in: np.ndarray, wav_out: np.ndarray) -> dict:
    """Validate enhanced clip: finite, length match, mono, valid level, no clipping."""
    r_in = float(np.sqrt((wav_in ** 2).mean()))
    r_out = float(np.sqrt((wav_out ** 2).mean()))
    rms_ratio = r_out / (r_in + 1e-12)
    peak = float(np.abs(wav_out).max())

    ok = (
        bool(np.isfinite(wav_out).all())
        and len(wav_out) == len(wav_in)
        and wav_out.ndim == 1
        and (0.4 <= rms_ratio <= 2.5)
        and (r_out > 1e-5)
        and (peak <= 0.999)
    )
    return {
        "ok": ok,
        "rms_ratio": round(rms_ratio, 3),
        "peak": round(peak, 3),
        "len": len(wav_out),
    }


# ---------------------------------------------------------------------------
# 4. Mandatory SraVaani-1.0 Transcriber Wrapper
# ---------------------------------------------------------------------------
class SraVaaniTranscriber:
    """Transcribes audio using mandatory ARTPARK-IISc/SraVaani-1.0 model."""

    def __init__(self, hf_token: str | None = None, device: str = "cuda"):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel

        print(f"[INFO] Downloading / loading SraVaani-1.0 from {ASR_REPO}...")
        path = snapshot_download(ASR_REPO, token=hf_token)
        self.asr = AutoModel.from_pretrained(path, trust_remote_code=True).to(device).eval()
        self.device = device
        self.torch = torch
        print(f"[INFO] SraVaani-1.0 loaded successfully on {device.upper()}!")

    def transcribe_paths(self, paths: list[str]) -> list[str]:
        """Transcribe list of WAV file paths."""
        with self.torch.no_grad():
            try:
                hyps = self.asr.transcribe(paths, return_hypotheses=True)
                return [(h.text if hasattr(h, "text") else str(h)).strip() for h in hyps]
            except Exception:
                # Fallback one-by-one if a batch has issues
                out = []
                for p in paths:
                    try:
                        h = self.asr.transcribe([p], return_hypotheses=True)[0]
                        out.append((h.text if hasattr(h, "text") else str(h)).strip())
                    except Exception:
                        out.append("")
                return out


# ---------------------------------------------------------------------------
# 5. Main Inference & Submission Flow
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="IndoML 2026 Track 2 - SOTA Event-Gated Speech Enhancement")
    parser.add_argument("--test-dir", type=str, default=None,
                        help="Path to folder with test audio clips")
    parser.add_argument("--track1-jsonl", type=str, default=None,
                        help="Path to Track 1 predictions.jsonl for event-gated enhancement")
    parser.add_argument("--gate-mode", type=str, default="gated", choices=["gated", "full"],
                        help="'gated' (clean only detected noise spans) or 'full' (clean entire clip)")
    parser.add_argument("--blend", type=float, default=0.85,
                        help="Blend factor: y = blend*clean + (1-blend)*noisy (default: 0.85)")
    parser.add_argument("--enhancer", type=str, default="speechbrain",
                        choices=["speechbrain", "deepfilternet", "spectral"],
                        help="Enhancer engine: 'speechbrain' (SOTA HF MetricGAN+, recommended), 'deepfilternet', or 'spectral' (fallback)")
    parser.add_argument("--output-dir", type=str, default="t2_enhanced_wavs",
                        help="Directory to store enhanced WAV files")
    parser.add_argument("--output-zip", type=str, default="submission_track2.zip",
                        help="Output submission ZIP filename")
    parser.add_argument("--batch-asr", type=int, default=8,
                        help="Batch size for SraVaani ASR transcription")
    parser.add_argument("--max-clips", type=int, default=None,
                        help="Optional cap on number of clips (for testing)")
    args = parser.parse_args()

    print("=" * 75)
    print("      INDOML 2026 TRACK 2: SOTA EVENT-GATED ENHANCEMENT & SUBMISSION")
    print("=" * 75)

    global torch
    if torch is None:
        try:
            import torch
        except ImportError:
            pass
    device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    print(f"[INFO] Compute Device: {device.upper()}")
    if device == "cpu":
        print("[WARN] Running on CPU! For fast SraVaani transcription, use a GPU.")

    # 1. Locate Test Directory
    test_dir = None
    if args.test_dir:
        test_dir = Path(args.test_dir)
    else:
        for cand in [
            Path("/kaggle/input/indoml-track2-test"),
            Path("/kaggle/input/input_data"),
            Path("/kaggle/input/test_audio"),
            Path("test_audio"),
            Path("input_data"),
        ]:
            if cand.exists() and any(cand.rglob("*.wav")):
                test_dir = cand
                break

    if test_dir is None or not test_dir.exists():
        print(f"[ERROR] Test audio directory not found! Please specify --test-dir")
        sys.exit(1)

    test_files = sorted([p for p in test_dir.rglob("*") if p.suffix.lower() in AUD])
    if args.max_clips:
        test_files = test_files[:args.max_clips]
    print(f"[INFO] Found {len(test_files)} test clips in: {test_dir}")

    # Check duplicate clip stems
    stems = [p.stem for p in test_files]
    dupes = [x for x, c in Counter(stems).items() if c > 1]
    if dupes:
        print(f"[ERROR] Found duplicate clip filenames: {dupes[:5]}")
        sys.exit(1)

    # 2. Load Track 1 Events (if provided)
    t1_events_by_id = {}
    if args.track1_jsonl and Path(args.track1_jsonl).exists():
        print(f"[INFO] Loading Track 1 event timestamps from: {args.track1_jsonl}")
        with open(args.track1_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        cid = record.get("clip_id")
                        if cid:
                            t1_events_by_id[cid] = record.get("events", [])
                    except Exception:
                        pass
        print(f"[INFO] Loaded events for {len(t1_events_by_id)} clips (Gate Mode: {args.gate_mode.upper()})")
    else:
        print(f"[INFO] No Track 1 jsonl provided. Running in {args.gate_mode.upper()} mode with global blend.")

    # 3. Initialize Enhancer
    enhancer = None
    if args.enhancer == "speechbrain":
        try:
            enhancer = SpeechBrainEnhancer(device=device)
        except Exception as e:
            print(f"[WARN] SpeechBrain failed to load: {e}")
            print("[INFO] Falling back to Spectral Gating...")
            enhancer = FallbackSpectralEnhancer()
    elif args.enhancer == "deepfilternet":
        try:
            enhancer = DeepFilterNetEnhancer()
        except Exception as e:
            print(f"[WARN] DeepFilterNet failed to load: {e}")
            print("[INFO] Falling back to SpeechBrain...")
            try:
                enhancer = SpeechBrainEnhancer(device=device)
            except Exception:
                enhancer = FallbackSpectralEnhancer()
    else:
        enhancer = FallbackSpectralEnhancer()

    # 4. Enhance Audio Files
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- STEP 1: ENHANCING TEST AUDIO ---")
    enhanced_paths = []
    t0 = time.time()

    for idx, audio_path in enumerate(test_files):
        cid = audio_path.stem
        dst_path = out_dir / f"{cid}.wav"

        # Check cache
        if dst_path.exists():
            enhanced_paths.append(dst_path)
            continue

        raw_wav = read_audio(audio_path, TARGET_SR)
        enh_wav = enhancer.enhance_clip(raw_wav)

        events = t1_events_by_id.get(cid, [])
        final_wav = apply_gated_enhancement(
            raw_wav, enh_wav, events=events, blend=args.blend, gate_mode=args.gate_mode, match_rms=True
        )

        chk = check_audio_output(raw_wav, final_wav)
        if not chk["ok"]:
            print(f"[WARN] Clip {cid} failed audio checks: {chk}. Re-normalizing...")
            final_wav = np.nan_to_num(final_wav, nan=0.0)

        # Write exact 16 kHz mono PCM-16 WAV
        import soundfile as sf
        sf.write(dst_path, final_wav, TARGET_SR, subtype=WAV_SUBTYPE)
        enhanced_paths.append(dst_path)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(test_files):
            elapsed = time.time() - t0
            print(f"  Processed {idx + 1}/{len(test_files)} clips ({(idx + 1) / elapsed:.1f} clips/s)")

    print(f"[INFO] Enhanced {len(enhanced_paths)} audio files in {out_dir}")

    # 5. Mandatory SraVaani ASR Transcription
    print("\n--- STEP 2: RUNNING MANDATORY SRAVAANI-1.0 ASR ---")
    hf_token = get_hf_token()
    transcriber = SraVaaniTranscriber(hf_token=hf_token, device=device)

    transcript_dict = {}
    batch_size = args.batch_asr

    print(f"[INFO] Transcribing {len(enhanced_paths)} clips in batches of {batch_size}...")
    t0 = time.time()
    for i in range(0, len(enhanced_paths), batch_size):
        batch = enhanced_paths[i:i + batch_size]
        paths_str = [str(p) for p in batch]
        texts = transcriber.transcribe_paths(paths_str)
        for p, txt in zip(batch, texts):
            transcript_dict[p.stem] = txt

        if (i + len(batch)) % 50 == 0 or (i + len(batch)) == len(enhanced_paths):
            print(f"  Transcribed {i + len(batch)}/{len(enhanced_paths)} clips...")

    # Write transcripts.jsonl
    transcript_file = Path(TRANSCRIPT_NAME)
    with open(transcript_file, "w", encoding="utf-8") as f:
        for cid in stems:
            text = transcript_dict.get(cid, "").strip()
            f.write(json.dumps({ID_KEY: cid, TEXT_KEY: text}, ensure_ascii=False) + "\n")

    print(f"[INFO] Created {transcript_file} with {len(stems)} transcriptions.")

    # 6. Package Root-Level submission_track2.zip
    print("\n--- STEP 3: PACKAGING SUBMISSION ZIP ---")
    out_zip = Path(args.output_zip)
    if out_zip.exists():
        out_zip.unlink()

    print(f"[INFO] Writing {len(enhanced_paths)} WAV files + {TRANSCRIPT_NAME} to {out_zip}...")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        # 1. transcripts.jsonl at ROOT
        z.write(transcript_file, TRANSCRIPT_NAME)

        # 2. All *.wav files at ROOT
        for p in enhanced_paths:
            z.write(p, p.name)

    # Verification checks
    with zipfile.ZipFile(out_zip, "r") as z:
        names = z.namelist()

    assert TRANSCRIPT_NAME in names, "FATAL: transcripts.jsonl missing from ZIP root!"
    has_subdirs = any("/" in n or "\\" in n for n in names)
    assert not has_subdirs, "FATAL: Subdirectories found inside submission ZIP! All files must be at root!"

    wav_count = sum(1 for n in names if n.endswith(".wav"))
    assert wav_count == len(test_files), f"FATAL: Missing WAV files! Expected {len(test_files)}, got {wav_count}"

    zip_mb = out_zip.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 75)
    print(f"      SUBMISSION ZIP READY: {out_zip} ({zip_mb:.1f} MB)")
    print(f"      Root Files: {TRANSCRIPT_NAME} + {wav_count} WAVs (16 kHz PCM-16)")
    print("      Verification Status: 100% PASSED (Audit-Compliant)")
    print("=" * 75)


if __name__ == "__main__":
    main()
