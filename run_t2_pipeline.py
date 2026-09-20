"""Track 2 Pipeline: Test Set Enhancement, Event-Gated Blending, SraVaani Transcription, and Packaging.

Loads the best trained MaskNet checkpoint (t2_masknet_best.pt),
applies Track 1 event gating to preserve 100% untouched clean speech outside noise events,
blends inside detected noise events, transcribes with ARTPARK-IISc/SraVaani-1.0 in batches,
and packages submission_track2.zip.
"""

import os
import sys
import glob
import json
import time
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import zipfile

from train_t2_champion import BiGRUMaskNet

ASR_REPO = "ARTPARK-IISc/SraVaani-1.0"

# ---------------------------------------------------------------------------
# SraVaani ASR Transcriber (Exact Audit-Compliant Batch Transcriber)
# ---------------------------------------------------------------------------
class SraVaaniTranscriber:
    """Transcribes audio using mandatory ARTPARK-IISc/SraVaani-1.0 model."""

    def __init__(self, hf_token=None, device="cuda"):
        from huggingface_hub import snapshot_download
        from transformers import AutoModel

        token = hf_token or os.environ.get("HF_TOKEN")
        print(f"[INFO] Downloading / loading SraVaani-1.0 from {ASR_REPO}...")
        path = snapshot_download(ASR_REPO, token=token)
        self.asr = AutoModel.from_pretrained(path, trust_remote_code=True).to(device).eval()
        self.device = device
        print(f"[INFO] SraVaani-1.0 loaded successfully on {device.upper()}!")

    def transcribe_paths(self, paths: list[str]) -> list[str]:
        """Transcribe list of WAV file paths in batches."""
        with torch.no_grad():
            try:
                hyps = self.asr.transcribe(paths, return_hypotheses=True)
                return [(h.text if hasattr(h, "text") else str(h)).strip() for h in hyps]
            except Exception:
                out = []
                for p in paths:
                    try:
                        h = self.asr.transcribe([p], return_hypotheses=True)[0]
                        out.append((h.text if hasattr(h, "text") else str(h)).strip())
                    except Exception:
                        out.append("")
                return out


# ---------------------------------------------------------------------------
# Mask Smoothing & Gating
# ---------------------------------------------------------------------------
def build_event_mask(duration_sec, events, sr=16000, win_samples=320):
    total_samples = int(round(duration_sec * sr))
    mask = np.zeros(total_samples, dtype=np.float32)
    for ev in events:
        s = max(0, int(round(ev["onset"] * sr)))
        e = min(total_samples, int(round(ev["offset"] * sr)))
        if e > s:
            mask[s:e] = 1.0
            
    if win_samples > 1 and mask.max() > 0:
        import scipy.signal
        kernel = np.hanning(win_samples)
        kernel /= kernel.sum()
        mask = scipy.signal.convolve(mask, kernel, mode="same")
        mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
        
    return mask


# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=str, default="/kaggle/working/test_audio/audio")
    parser.add_argument("--track1-jsonl", type=str, default="/kaggle/working/indoml-2026/predictions.jsonl")
    parser.add_argument("--ckpt", type=str, default="/kaggle/working/t2_masknet_best.pt")
    parser.add_argument("--blend", type=float, default=0.80, help="Weight on enhanced speech inside noise events")
    parser.add_argument("--out-dir", type=str, default="/kaggle/working/t2_champion_wavs")
    parser.add_argument("--out-zip", type=str, default="/kaggle/working/submission_track2.zip")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1. Load Trained MaskNet Checkpoint
    assert os.path.exists(args.ckpt), f"Checkpoint not found at {args.ckpt}"
    ckpt_data = torch.load(args.ckpt, map_location=device)
    model = BiGRUMaskNet().to(device)
    model.load_state_dict(ckpt_data["model_state_dict"])
    model.eval()
    print(f"[INFO] Loaded MaskNet checkpoint from {args.ckpt} (SI-SDR: {ckpt_data.get('val_si_sdr', 'N/A'):.2f} dB)")

    # 2. Load Track 1 Event Predictions
    t1_events = {}
    if os.path.exists(args.track1_jsonl):
        with open(args.track1_jsonl) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    t1_events[rec["clip_id"]] = rec.get("events", [])
        print(f"[INFO] Loaded Track 1 events for {len(t1_events)} clips.")
    else:
        print(f"[WARN] Track 1 predictions not found at {args.track1_jsonl}. Will enhance all spans.")

    # 3. Gather Test Audio Files
    test_files = sorted(glob.glob(os.path.join(args.test_dir, "*.wav")))
    print(f"[INFO] Found {len(test_files)} test audio files in {args.test_dir}")
    assert len(test_files) > 0, "No test files found!"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4. Enhance all clips
    print("\n[INFO] Phase 1/3: Running Event-Gated Speech Enhancement...")
    t0 = time.time()
    enhanced_wav_paths = []

    for fpath in tqdm(test_files, desc="Enhancing WAVs"):
        fname = os.path.basename(fpath)
        cid = fname[:-4] if fname.lower().endswith(".wav") else fname

        raw_wav, sr = sf.read(fpath, dtype="float32")
        events = t1_events.get(cid, [])

        if events:
            # Run MaskNet inference on GPU
            with torch.no_grad():
                inp = torch.from_numpy(raw_wav).unsqueeze(0).to(device)
                est_wav, _ = model(inp)
                enh_wav = est_wav[0].cpu().numpy()

            dur = len(raw_wav) / 16000.0
            mask = build_event_mask(dur, events, sr=16000)
            if len(mask) < len(raw_wav):
                mask = np.pad(mask, (0, len(raw_wav) - len(mask)))
            mask = mask[:len(raw_wav)]

            # Gated Blend: outside events = 100% clean raw_wav; inside events = blend*enh + (1-blend)*raw
            final_wav = (1.0 - mask) * raw_wav + mask * (args.blend * enh_wav + (1.0 - args.blend) * raw_wav)
        else:
            # Clean speech detected -> leave untouched!
            final_wav = raw_wav

        # Peak normalization guard
        pk = np.abs(final_wav).max()
        if pk > 0.99:
            final_wav = final_wav / pk * 0.99

        # Save 16 kHz PCM-16 WAV
        out_wav_path = out_dir / fname
        sf.write(out_wav_path, final_wav, 16000, subtype="PCM_16")
        enhanced_wav_paths.append(str(out_wav_path))

    enh_time = time.time() - t0
    print(f"[INFO] Enhanced {len(test_files)} clips in {enh_time/60:.2f} minutes.")

    # 5. Load SraVaani Transcriber & Transcribe
    print("\n[INFO] Phase 2/3: Transcribing with SraVaani-1.0 (Audit Compliance)...")
    transcriber = SraVaaniTranscriber(hf_token=args.hf_token, device=str(device))

    transcripts = {}
    bs = args.batch_size
    t1 = time.time()

    for i in tqdm(range(0, len(enhanced_wav_paths), bs), desc="Transcribing Batches"):
        batch_paths = enhanced_wav_paths[i : i + bs]
        texts = transcriber.transcribe_paths(batch_paths)
        for p, txt in zip(batch_paths, texts):
            cid = os.path.basename(p)[:-4]
            transcripts[cid] = txt

    asr_time = time.time() - t1
    print(f"[INFO] Transcribed {len(transcripts)} clips in {asr_time/60:.2f} minutes.")

    # Write transcripts.jsonl
    transcripts_path = out_dir / "transcripts.jsonl"
    with open(transcripts_path, "w", encoding="utf-8") as f:
        for fpath in test_files:
            fname = os.path.basename(fpath)
            cid = fname[:-4] if fname.lower().endswith(".wav") else fname
            text = transcripts.get(cid, "")
            line = json.dumps({"clip_id": cid, "text": text}, ensure_ascii=False)
            f.write(line + "\n")

    print(f"[INFO] Saved transcripts.jsonl ({len(transcripts)} lines).")

    # 6. Package submission_track2.zip
    print(f"\n[INFO] Phase 3/3: Packaging {args.out_zip} ...")
    with zipfile.ZipFile(args.out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(transcripts_path, arcname="transcripts.jsonl")
        for fpath in tqdm(test_files, desc="Zipping WAVs"):
            fname = os.path.basename(fpath)
            z.write(out_dir / fname, arcname=fname)

    zip_size_mb = os.path.getsize(args.out_zip) / (1024 * 1024)
    print(f"\n[SUCCESS] Final Submission Ready: {args.out_zip} ({zip_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
