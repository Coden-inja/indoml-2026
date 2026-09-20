"""Track 2 Pipeline: Test Set Enhancement, ASR Transcription, and Submission Packaging.

Loads the best trained MaskNet checkpoint, applies Track 1 event gating to preserve
clean speech intervals, blends inside detected noise spans, transcribes with SraVaani-1.0,
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

from train_t2_champion import BiGRUMaskNet, calc_si_sdr_np

# ---------------------------------------------------------------------------
# SraVaani ASR Transcriber
# ---------------------------------------------------------------------------
class SraVaaniTranscriber:
    def __init__(self, token=None, device="cuda"):
        from transformers import AutoModelForCTC, AutoProcessor
        token = token or os.environ.get("HF_TOKEN")
        print("[INFO] Loading ARTPARK-IISc/SraVaani-1.0 ASR model...")
        self.processor = AutoProcessor.from_pretrained("ARTPARK-IISc/SraVaani-1.0", token=token)
        self.model = AutoModelForCTC.from_pretrained(
            "ARTPARK-IISc/SraVaani-1.0",
            token=token,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device).eval()
        self.device = device
        print("[INFO] SraVaani-1.0 loaded successfully.")

    @torch.no_grad()
    def transcribe(self, audio_16k):
        inputs = self.processor(audio_16k, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to(self.device)
        if self.device == "cuda":
            input_values = input_values.half()
        logits = self.model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        text = self.processor.batch_decode(pred_ids)[0]
        return text.strip()


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
    parser.add_argument("--blend", type=float, default=0.75, help="Weight on enhanced speech inside noise events")
    parser.add_argument("--out-dir", type=str, default="/kaggle/working/t2_champion_wavs")
    parser.add_argument("--out-zip", type=str, default="/kaggle/working/submission_track2.zip")
    parser.add_argument("--hf-token", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1. Load Trained MaskNet
    assert os.path.exists(args.ckpt), f"Checkpoint not found at {args.ckpt}"
    ckpt_data = torch.load(args.ckpt, map_location=device)
    model = BiGRUMaskNet().to(device)
    model.load_state_dict(ckpt_data["model_state_dict"])
    model.eval()
    print(f"[INFO] Loaded MaskNet checkpoint from {args.ckpt} (SI-SDR: {ckpt_data.get('val_si_sdr', 'N/A')})")

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

    # 3. Load SraVaani ASR
    transcriber = SraVaaniTranscriber(token=args.hf_token, device=str(device))

    # 4. Gather Test Audio Files
    test_files = sorted(glob.glob(os.path.join(args.test_dir, "*.wav")))
    print(f"[INFO] Found {len(test_files)} test audio files in {args.test_dir}")
    assert len(test_files) > 0, "No test files found!"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transcripts = {}
    t0 = time.time()
    print("\n[INFO] Starting Enhancement + Gating + Transcription...")

    for fpath in tqdm(test_files, desc="Enhancing & Transcribing"):
        fname = os.path.basename(fpath)
        cid = fname[:-4] if fname.lower().endswith(".wav") else fname

        raw_wav, sr = sf.read(fpath, dtype="float32")
        assert sr == 16000, f"Expected 16kHz, got {sr}"

        # Run MaskNet inference
        with torch.no_grad():
            inp = torch.from_numpy(raw_wav).unsqueeze(0).to(device)
            est_wav, _ = model(inp)
            enh_wav = est_wav[0].cpu().numpy()

        # Build Track 1 Event Mask
        events = t1_events.get(cid, [])
        if events:
            dur = len(raw_wav) / 16000.0
            mask = build_event_mask(dur, events, sr=16000)
            if len(mask) < len(raw_wav):
                mask = np.pad(mask, (0, len(raw_wav) - len(mask)))
            mask = mask[:len(raw_wav)]
            
            # Gated Blend:
            # Outside events: raw_wav (100% clean identity)
            # Inside events: blend * enh_wav + (1 - blend) * raw_wav
            final_wav = (1.0 - mask) * raw_wav + mask * (args.blend * enh_wav + (1.0 - args.blend) * raw_wav)
        else:
            # No events detected -> leave 100% untouched raw audio
            final_wav = raw_wav

        # Peak normalization guard
        pk = np.abs(final_wav).max()
        if pk > 0.99:
            final_wav = final_wav / pk * 0.99

        # Save 16 kHz PCM-16 WAV
        out_wav_path = out_dir / fname
        sf.write(out_wav_path, final_wav, 16000, subtype="PCM_16")

        # Transcribe with SraVaani
        text = transcriber.transcribe(final_wav)
        transcripts[cid] = text

    elapsed = time.time() - t0
    print(f"\n[INFO] Finished enhancing and transcribing {len(test_files)} clips in {elapsed/60:.2f} minutes.")

    # 5. Write transcripts.jsonl
    transcripts_path = out_dir / "transcripts.jsonl"
    with open(transcripts_path, "w", encoding="utf-8") as f:
        for fpath in test_files:
            fname = os.path.basename(fpath)
            cid = fname[:-4] if fname.lower().endswith(".wav") else fname
            text = transcripts.get(cid, "")
            line = json.dumps({"clip_id": cid, "text": text}, ensure_ascii=False)
            f.write(line + "\n")

    print(f"[INFO] Saved transcripts to {transcripts_path} ({len(transcripts)} entries).")

    # 6. Package submission_track2.zip
    print(f"[INFO] Packaging {args.out_zip} ...")
    with zipfile.ZipFile(args.out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        # Add transcripts.jsonl at root
        z.write(transcripts_path, arcname="transcripts.jsonl")
        for fpath in tqdm(test_files, desc="Zipping WAVs"):
            fname = os.path.basename(fpath)
            z.write(out_dir / fname, arcname=fname)

    zip_size_mb = os.path.getsize(args.out_zip) / (1024 * 1024)
    print(f"[SUCCESS] Submission packaged: {args.out_zip} ({zip_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
