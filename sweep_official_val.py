"""Official Validation Sweep on CodaBench Validation Dataset.

Loads /content/val_data/validation/validationMetadata.json (2,501 clips with exact GT timestamps)
Runs model inference ONCE over the validation audio to cache posteriors,
then executes an exhaustive grid sweep to find the global optimum parameters (thr, med, min_dur, merge_gap, gate).
Finally, runs inference on /content/test_audio with the winning parameters to generate `submission_track1_optimal.zip`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from pathlib import Path

# Add modular package to python path
pkg_dir = Path(__file__).resolve().parent / "track1_detection" / "versions" / "v-3_indoml2026_track1" / "modular"
sys.path.insert(0, str(pkg_dir))

import torch
import numpy as np
from tqdm.auto import tqdm

from config import CFG, FRAME_SEC, MIN_SAMPLES
AUD = (".wav", ".flac", ".mp3", ".ogg")
from model import WavLMSED
from inference import t1_posteriors
from evaluate import (
    prob_to_events, event_based_f1, segment_dice
)
from predict_track1 import read_audio, export_track1_tuned


def is_valid_torch_file(p: Path) -> bool:
    if not p.is_file():
        return False
    try:
        import zipfile
        with zipfile.ZipFile(str(p), "r") as z:
            names = z.namelist()
            return any("data.pkl" in n for n in names) and any("version" in n for n in names)
    except Exception:
        return False


def parse_ground_truth(meta_path: Path):
    """Parse exact ground truth intervals from validationMetadata.json."""
    with open(meta_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    gt_dict = {}
    for item in items:
        fn = item.get("segmentFileName", "")
        cid = fn[:-4] if fn.lower().endswith(".wav") else fn
        events = []
        raw_events = item.get("NoiseSubCategoryTimeStamp", []) or []
        for ev in raw_events:
            try:
                s = float(ev.get("start", 0.0))
                e = float(ev.get("end", 0.0))
                if e > s:
                    events.append((round(s, 3), round(e, 3)))
            except (ValueError, TypeError):
                continue
        gt_dict[cid] = events
    return gt_dict


def main():
    parser = argparse.ArgumentParser(description="Official CodaBench Validation Sweep & Optimal Predictor")
    parser.add_argument("--ckpt", type=str, default="/content/t1_wavlm.pt", help="Path to checkpoint")
    parser.add_argument("--val-dir", type=str, default="/content/val_data/validation", help="Path to validation folder")
    parser.add_argument("--num-eval", type=int, default=300, help="Number of validation clips to evaluate (default 300 for speed, 0 for all)")
    parser.add_argument("--test-dir", type=str, default="/content/test_audio", help="Path to test audio folder")
    parser.add_argument("--output-zip", type=str, default="submission_track1_optimal.zip", help="Output zip name")
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

    print("=" * 80)
    print("      OFFICIAL CODABENCH VALIDATION SWEEP & OPTIMAL SUBMISSION GENERATOR")
    print("=" * 80)

    # 1. Parse Ground Truth
    gt_dict = parse_ground_truth(meta_path)
    print(f"[INFO] Parsed ground-truth events for {len(gt_dict):,} validation clips.")

    # 2. Locate audio files
    audio_files = {}
    for p in val_dir.rglob("*"):
        if p.suffix.lower() in AUD:
            audio_files[p.stem] = p
    print(f"[INFO] Located {len(audio_files):,} validation WAV files on disk.")

    matched_cids = [cid for cid in gt_dict if cid in audio_files]
    print(f"[INFO] {len(matched_cids):,} clips have both audio and ground-truth labels.")

    if args.num_eval > 0 and args.num_eval < len(matched_cids):
        # Sample deterministically
        import random
        random.seed(42)
        matched_cids = random.sample(matched_cids, args.num_eval)
        print(f"[INFO] Subsetting to {len(matched_cids)} clips for fast empirical sweep.")

    # 3. Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Compute device: {device.upper()}")

    # Find checkpoint
    ckpt_path = Path(args.ckpt)
    for cand in [
        ckpt_path,
        Path("/content") / ckpt_path.name,
        Path("/content/t1_wavlm.pt"),
        Path("/content/t1_wavlm.zip"),
        Path("/kaggle/working/t1_wavlm.pt"),
        Path("t1_wavlm.pt"),
    ]:
        if cand.exists() and cand.is_file() and is_valid_torch_file(cand):
            ckpt_path = cand
            break

    # If checkpoint is extracted as a directory
    if not ckpt_path.exists() or ckpt_path.is_dir() or not is_valid_torch_file(ckpt_path):
        for input_root in [Path("/kaggle/input"), Path("/content")]:
            if input_root.exists():
                data_pkls = [p for p in input_root.rglob("data.pkl") if "sample_data" not in str(p)]
                if data_pkls:
                    model_dir = data_pkls[0].parent
                    print(f"[INFO] Detected unzipped checkpoint directory at: {model_dir}")
                    print(f"[INFO] Packaging into t1_wavlm.pt...")
                    import shutil
                    work_dir = Path("/content") if Path("/content").exists() else Path("/kaggle/working")
                    cand_zip = work_dir / "t1_wavlm.zip"
                    cand_pt = work_dir / "t1_wavlm.pt"
                    if cand_pt.exists():
                        cand_pt.unlink()
                    if cand_zip.exists():
                        cand_zip.unlink()
                    shutil.make_archive(str(work_dir / "t1_wavlm"), "zip", root_dir=str(model_dir.parent), base_dir=model_dir.name)
                    if cand_zip.exists():
                        cand_zip.rename(cand_pt)
                    ckpt_path = cand_pt
                    break

    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path.resolve()}")
        return

    print(f"[INFO] Loading checkpoint: {ckpt_path.resolve()}")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = WavLMSED().to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    # 4. Run Model ONCE & Cache Posteriors (or load from disk)
    cache_file = Path(f"val_cache_{len(matched_cids)}.pt")
    if cache_file.exists():
        print(f"\n--- 1. LOADING CACHED POSTERIORS FROM DISK ({cache_file.name}) ---")
        saved = torch.load(cache_file, map_location="cpu", weights_only=False)
        cache = saved["cache"]
        ref_dict = saved["ref_dict"]
        print(f"[SUCCESS] Loaded {len(cache)} cached validation clip posteriors.")
    else:
        print("\n--- 1. RUNNING MODEL FORWARD PASS ON VALIDATION CLIPS ---")
        cache = []
        ref_dict = {}

        for cid in tqdm(matched_cids, desc="Caching Posteriors"):
            wav_path = audio_files[cid]
            wav = read_audio(wav_path, sr=CFG["sr"])
            dur = len(wav) / CFG["sr"]
            post, cp = t1_posteriors(wav, model, device, sr=CFG["sr"])
            cache.append({
                "cid": cid,
                "post": post[0].copy(),
                "cp": float(cp),
                "dur": float(dur),
            })
            ref_dict[cid] = gt_dict[cid]

        print(f"[SUCCESS] Cached posterior maps for {len(cache)} validation clips. Saving to {cache_file}...")
        torch.save({"cache": cache, "ref_dict": ref_dict}, cache_file)

    # Calculate baseline score for comparison
    def eval_config(thr, med, mind, mgap, gate):
        pred_dict = {}
        for item in cache:
            cid, dur = item["cid"], item["dur"]
            if item["cp"] < gate:
                pred_dict[cid] = []
            else:
                raw_evs = prob_to_events(item["post"], thr=thr, med=med, min_dur=mind, merge_gap=mgap)
                valid_evs = []
                for on, off in raw_evs:
                    on, off = max(0.0, on), min(dur, off)
                    if off - on >= mind:
                        valid_evs.append((round(float(on), 3), round(float(off), 3)))
                pred_dict[cid] = valid_evs
        f1, prec, rec, tp, fp, fn = event_based_f1(ref_dict, pred_dict)
        dice = segment_dice(ref_dict, pred_dict)
        return {"comb": f1 + dice, "f1": f1, "dice": dice, "prec": prec, "rec": rec}

    base_metrics = eval_config(thr=0.45, med=7, mind=0.05, mgap=0.05, gate=0.25)
    print("\n" + "=" * 80)
    print(f"  BASELINE (Submission #1 config: thr=0.45, med=7, mind=0.05s, mgap=0.05s, gate=0.25)")
    print(f"  Validation Comb: {base_metrics['comb']:.4f} | Event F1: {base_metrics['f1']:.4f} | Segment Dice: {base_metrics['dice']:.4f}")
    print("=" * 80)

    # 5. Grid Search over Post-Processing Hyperparameters
    print("\n--- 2. EXHAUSTIVE EMPIRICAL HYPERPARAMETER SWEEP ---")
    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    medians = [5, 7, 9, 11, 13, 15]
    min_durs = [0.03, 0.05, 0.08, 0.10]
    merge_gaps = [0.03, 0.05, 0.08, 0.10]
    clip_gates = [0.0, 0.10, 0.20, 0.30]

    total_configs = len(thresholds) * len(medians) * len(min_durs) * len(merge_gaps) * len(clip_gates)
    print(f"[INFO] Scoring {total_configs:,} parameter combinations against real ground truth...")

    results = []
    for thr in thresholds:
        for med in medians:
            for mind in min_durs:
                for mgap in merge_gaps:
                    for gate in clip_gates:
                        m = eval_config(thr, med, mind, mgap, gate)
                        results.append({
                            "comb": float(m["comb"]),
                            "f1": float(m["f1"]),
                            "dice": float(m["dice"]),
                            "prec": float(m["prec"]),
                            "rec": float(m["rec"]),
                            "thr": float(thr),
                            "med": int(med),
                            "mind": float(mind),
                            "mgap": float(mgap),
                            "gate": float(gate),
                        })

    results.sort(key=lambda x: x["comb"], reverse=True)

    print("\n" + "=" * 80)
    print("           TOP 10 WINNING CONFIGURATIONS ON OFFICIAL VALIDATION SET")
    print("=" * 80)
    print(f"{'Rank':<5}{'Comb':>8}{'F1':>8}{'Dice':>8}{'thr':>6}{'med':>5}{'mind':>7}{'mgap':>7}{'gate':>6}{'P':>7}{'R':>7}")
    print("-" * 80)
    for i, r in enumerate(results[:10]):
        print(f"#{i+1:<4}{r['comb']:8.4f}{r['f1']:8.3f}{r['dice']:8.3f}{r['thr']:6.2f}{r['med']:5d}{r['mind']:7.2f}{r['mgap']:7.2f}{r['gate']:6.2f}{r['prec']:7.3f}{r['rec']:7.3f}")
    print("=" * 80)

    best = results[0]
    print(f"\n[WINNER] Highest Validation Combined Score: {best['comb']:.4f}")
    print(f"         Event F1: {best['f1']:.4f} | Segment Dice: {best['dice']:.4f}")
    print(f"         Precision: {best['prec']:.4f} | Recall: {best['rec']:.4f}")
    print(f"         Optimal Parameters: thr={best['thr']}, med={best['med']}, min_dur={best['mind']}s, merge_gap={best['mgap']}s, gate={best['gate']}")

    # 6. Run Test Inference using Proven Winning Parameters
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
    print(f"\n--- 3. GENERATING OPTIMAL TEST SUBMISSION ({len(files):,} clips) ---")
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

    print("\n" + "=" * 80)
    print("                   OPTIMAL SUBMISSION READY!")
    print("=" * 80)
    print(f"  Output ZIP:      {out_zip.resolve()} ({out_zip.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  Validation Comb: {best['comb']:.4f} (F1: {best['f1']:.4f}, Dice: {best['dice']:.4f})")
    print(f"  Winning Config:  thr={best['thr']}, med={best['med']}, min_dur={best['mind']}, mgap={best['mgap']}, gate={best['gate']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
