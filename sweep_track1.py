"""Offline Validation Sweep & Optimal Submission Generator for Track 1.

Runs an empirical grid search over held-out district validation clips to find
the exact (thr, med, min_dur, gate, merge_gap) that maximizes Combined Score (F1 + Dice).
Then immediately produces `submission_track1_optimal.zip` using the winning parameters.

Usage on Colab or Kaggle:
  python sweep_track1.py --val-clips 100 --test-dir /content/test_audio
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import zipfile
from pathlib import Path

# Add modular package to python path
pkg_dir = Path(__file__).resolve().parent / "track1_detection" / "versions" / "v-3_indoml2026_track1" / "modular"
sys.path.insert(0, str(pkg_dir))

import torch
import numpy as np
from tqdm.auto import tqdm

from config import CFG, WORK, FRAME_SEC, MIN_SAMPLES
AUD = (".wav", ".flac", ".mp3", ".ogg")
import data
from model import WavLMSED
from inference import t1_posteriors
from evaluate import (
    cache_faithful_posteriors, score_faithful_cached, prob_to_events,
    event_based_f1, segment_dice
)
from predict_track1 import read_audio, export_track1_tuned


def find_checkpoint(user_ckpt: str = "t1_wavlm.pt") -> Path:
    """Find and validate checkpoint file across candidate paths."""
    cand_p = Path(user_ckpt)
    candidates = [
        cand_p,
        Path("/content") / cand_p.name,
        Path("/content/t1_wavlm.pt"),
        Path("/content/t1_wavlm.zip"),
        Path("/kaggle/working") / cand_p.name,
        Path("/kaggle/working/t1_wavlm.pt"),
        Path("t1_wavlm.pt"),
        Path("t1_wavlm.zip"),
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c

    # Search recursively
    for root_dir in [Path("/content"), Path("/kaggle"), Path(".")]:
        if root_dir.exists():
            for f in root_dir.rglob("t1_wavlm*"):
                if f.is_file() and f.suffix in [".pt", ".zip"]:
                    return f

    raise FileNotFoundError(f"Could not locate t1_wavlm checkpoint! Checked: {candidates}")


def main():
    parser = argparse.ArgumentParser(description="Track 1 Empirical Validation Sweep & Predictor")
    parser.add_argument("--ckpt", type=str, default="t1_wavlm.pt", help="Path to trained checkpoint")
    parser.add_argument("--val-clips", type=int, default=150, help="Number of held-out validation clips to evaluate")
    parser.add_argument("--test-dir", type=str, default="/content/test_audio", help="Path to test audio directory")
    parser.add_argument("--output-zip", type=str, default="submission_track1_optimal.zip", help="Output zip name")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 75)
    print("       INDOML 2026 TRACK 1: EMPIRICAL VALIDATION SWEEP & OPTIMIZER")
    print(f"       Device: {device.upper()} | Held-out Val Clips: {args.val_clips}")
    print("=" * 75)

    ckpt_path = find_checkpoint(args.ckpt)
    print(f"[INFO] Loading checkpoint: {ckpt_path.resolve()}")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = WavLMSED().to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    # 1. Stream held-out validation data from HuggingFace
    print("\n--- 1. FETCHING HELD-OUT DISTRICT VALIDATION SET ---")
    hf_token = os.environ.get("HF_TOKEN")
    raw_stream = data.load_stream(hf_token)
    # Target district-disjoint holdouts
    val_districts = ["Bhopal", "Unakoti", "Katni", "Dhar"]
    print(f"[INFO] Filtering Gold-tier clips from held-out districts: {val_districts}...")

    val_rows = []
    for item in raw_stream:
        props = item.get("properties", {}) or {}
        ann = props.get("annotationQuality", "")
        dist = props.get("district", "")
        if ann == "verified_timestamps" and dist in val_districts:
            w_int16 = data.decode_audio(item)
            if w_int16 is not None:
                spans = data.extract_gold_spans(props)
                val_rows.append({
                    "clip_id": props.get("clip_id") or f"val_{len(val_rows)}",
                    "wav": w_int16,
                    "raw_spans": spans,
                    "district": dist,
                })
                if len(val_rows) >= args.val_clips:
                    break

    print(f"[SUCCESS] Collected {len(val_rows)} faithful validation clips with ground truth!")

    # 2. Cache posteriors ONCE
    print("\n--- 2. RUNNING MODEL FORWARD PASS ON VALIDATION SET ---")
    cache = cache_faithful_posteriors(model, val_rows, device)
    print(f"[SUCCESS] Cached {len(cache)} whole-clip posterior maps.")

    # 3. Exhaustive Grid Search
    print("\n--- 3. SWEEPING POST-PROCESSING HYPERPARAMETER GRID ---")
    sweep_results = []
    
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    medians = [3, 5, 7, 9]
    min_durs = [0.03, 0.05, 0.08, 0.10]
    merge_gaps = [0.03, 0.05, 0.08, 0.10]
    clip_gates = [0.0, 0.15, 0.25]

    total_configs = len(thresholds) * len(medians) * len(min_durs) * len(merge_gaps) * len(clip_gates)
    print(f"[INFO] Evaluating {total_configs:,} parameter combinations...")

    for thr in thresholds:
        for med in medians:
            for mind in min_durs:
                for mgap in merge_gaps:
                    for gate in clip_gates:
                        m = score_faithful_cached(
                            cache, thr=thr, med=med, min_dur=mind, merge_gap=mgap, clip_gate=gate
                        )
                        sweep_results.append({
                            "comb": float(m["score"]),
                            "f1": float(m["f1"]),
                            "dice": float(m["dice"]),
                            "prec": float(m["precision"]),
                            "rec": float(m["recall"]),
                            "thr": float(thr),
                            "med": int(med),
                            "mind": float(mind),
                            "mgap": float(mgap),
                            "gate": float(gate),
                        })

    sweep_results.sort(key=lambda x: x["comb"], reverse=True)

    print("\n" + "=" * 80)
    print("                     TOP 10 VALIDATION CONFIGURATIONS")
    print("=" * 80)
    print(f"{'Rank':<5}{'Comb':>8}{'F1':>8}{'Dice':>8}{'thr':>6}{'med':>5}{'mind':>7}{'mgap':>7}{'gate':>6}{'P':>7}{'R':>7}")
    print("-" * 80)
    for i, r in enumerate(sweep_results[:10]):
        print(f"#{i+1:<4}{r['comb']:8.4f}{r['f1']:8.3f}{r['dice']:8.3f}{r['thr']:6.2f}{r['med']:5d}{r['mind']:7.2f}{r['mgap']:7.2f}{r['gate']:6.2f}{r['prec']:7.3f}{r['rec']:7.3f}")
    print("=" * 80)

    best = sweep_results[0]
    print(f"\n[WINNER] Highest Validation Combined Score: {best['comb']:.4f}")
    print(f"         Event F1: {best['f1']:.4f} | Segment Dice: {best['dice']:.4f}")
    print(f"         Optimal Parameters: thr={best['thr']}, med={best['med']}, min_dur={best['mind']}s, merge_gap={best['mgap']}s, gate={best['gate']}")

    # 4. Generate Test Predictions with Winning Parameters
    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        for cand in [Path("/content/test_audio"), Path("/kaggle/working/test_audio"), Path("test_audio")]:
            if cand.exists():
                test_dir = cand
                break

    if not test_dir.exists():
        print(f"\n[WARN] Test directory '{test_dir}' not found. Skipping submission zip generation.")
        return

    files = sorted([p for p in test_dir.rglob("*") if p.suffix.lower() in AUD])
    print(f"\n--- 4. GENERATING OPTIMAL TEST SUBMISSION ({len(files):,} clips) ---")
    jsonl_path = Path("predictions.jsonl")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for p in tqdm(files, desc="Optimal Inference"):
            wav = read_audio(p, sr=CFG["sr"])
            rec = export_track1_tuned(
                p.stem, wav, model, device,
                thr=best["thr"], med=best["med"], min_dur=best["mind"],
                gate=best["gate"], merge_gap=best["mgap"]
            )
            f.write(json.dumps({"clip_id": p.stem, "events": rec["events"]}, ensure_ascii=False) + "\n")

    out_zip = Path(args.output_zip)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(jsonl_path, "predictions.jsonl")

    print("\n" + "=" * 75)
    print("               OPTIMAL SUBMISSION READY!")
    print("=" * 75)
    print(f"  Output ZIP: {out_zip.resolve()} ({out_zip.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  Backed by {total_configs:,} empirical validation evaluations.")
    print("=" * 75)


if __name__ == "__main__":
    main()
