"""Track 1 Training Runner with District-Disjoint Validation.

This script launches the 2-stage WavLM SED baseline with:
  1. Zero speaker leakage: Held-out Gold districts for validation.
  2. Whole-clip faithful scoring: Evaluates against uncropped raw ground-truth spans.
  3. Silver-tier whole-clip downweighting: Neutralizes lazy full-clip annotations.
  4. Post-processing optimization: Sweeps threshold, median filter, and min duration.

Usage on Kaggle:
  # Quick validation run (~20-30 mins on GPU):
  !python run_train_track1.py --budget fast --split district

  # Full training run (~4-6 hours on GPU):
  !python run_train_track1.py --budget full --split district
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add modular package to python path
pkg_dir = Path(__file__).resolve().parent / "track1_detection" / "versions" / "v-3_indoml2026_track1" / "modular"
sys.path.insert(0, str(pkg_dir))

import torch
from config import BUDGET, CFG, CKPT, WORK, budget_settings, seed_everything
import data
from model import WavLMSED
from train import train
from evaluate import cache_faithful_posteriors, sweep_faithful_postprocessing


def get_hf_token() -> str | None:
    """Retrieve HF_TOKEN from Kaggle secrets or environment."""
    try:
        from kaggle_secrets import UserSecretsClient
        tok = UserSecretsClient().get_secret("HF_TOKEN")
        if tok:
            return tok
    except Exception:
        pass
    return os.environ.get("HF_TOKEN")


def main():
    parser = argparse.ArgumentParser(description="IndoML 2026 Track 1 - WavLM SED Training")
    parser.add_argument("--budget", type=str, default="fast", choices=["fast", "full"],
                        help="Budget profile: 'fast' (smoke/dev) or 'full' (leaderboard run)")
    parser.add_argument("--split", type=str, default="district", choices=["district", "random"],
                        help="Validation split strategy: 'district' (zero speaker leakage) or 'random'")
    parser.add_argument("--val-districts", nargs="+", default=["Bhopal", "Unakoti", "Katni", "Dhar"],
                        help="Districts to hold out for validation when using district split")
    parser.add_argument("--train-limit", type=int, default=None,
                        help="Optional override to limit number of training clips")
    parser.add_argument("--output-ckpt", type=str, default="t1_wavlm_best.pt",
                        help="Path to save best trained checkpoint")
    args = parser.parse_args()

    print("=" * 70)
    print("      INDOML 2026 TRACK 1: WAVLM SED TRAINING & FAITHFUL EVALUATION")
    print(f"      Budget: {args.budget} | Split: {args.split}")
    print("=" * 70)

    seed_everything(CFG["seed"])
    data.seed_torch(CFG["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Running on compute device: {device.upper()}")
    if device == "cpu":
        print("[WARN] Running on CPU! For real training, switch Kaggle Accelerator to GPU (T4 or P100).")

    hf_token = get_hf_token()
    max_per_quality, stage1_epochs, stage2_epochs, t1_bs, default_limit = budget_settings(args.budget)
    train_limit = args.train_limit if args.train_limit is not None else default_limit

    print(f"[INFO] Budget Settings: Stage1={stage1_epochs} eps, Stage2={stage2_epochs} eps, BatchSize={t1_bs}")
    print(f"[INFO] Tier Caps: {max_per_quality} | Train Limit: {train_limit}")

    # 1. Load stream and build rows with whole-clip and district tracking
    print("\n--- 1. STREAMING & DECODING DATASET ---")
    raw_stream = data.load_stream(hf_token)
    rows = data.build_rows(raw_stream, max_per_quality)

    # 2. Bucket tiers and build district-disjoint split
    print("\n--- 2. CREATING SPLIT & DATALOADERS ---")
    gold, silver, bronze = data.bucket_by_tier(rows)
    train_dl, val_dl, val_rows = data.split_and_load(
        gold, silver, bronze, t1_bs,
        train_limit=train_limit,
        split_by=args.split,
        val_districts=args.val_districts,
    )

    # 3. Two-Stage Mean-Teacher Training
    print("\n--- 3. STARTING TWO-STAGE MEAN-TEACHER TRAINING ---")
    state = train(train_dl, val_dl, device, stage1_epochs, stage2_epochs)

    # 4. Faithful Whole-Clip Validation Evaluation
    print("\n--- 4. FAITHFUL WHOLE-CLIP VALIDATION EVALUATION ---")
    ckpt_path = Path(args.output_ckpt)
    # Load best checkpoint saved by train.py
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    model = WavLMSED().to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    print(f"[INFO] Running whole-clip inference over {len(val_rows)} validation clips...")
    cache = cache_faithful_posteriors(model, val_rows, device)
    (best_thr, best_med, best_min_dur, best_gate), best_result = sweep_faithful_postprocessing(cache, verbose=True)

    print("\n" + "=" * 70)
    print("              FINAL FAITHFUL VALIDATION RESULTS")
    print("=" * 70)
    print(f"  Combined Score:  {best_result[0]:.4f} / 2.0000")
    print(f"  Event F1:        {best_result[1]:.4f}  (Precision: {best_result[7]:.4f}, Recall: {best_result[8]:.4f})")
    print(f"  Segment Dice:    {best_result[2]:.4f}")
    print("  Optimal Post-Processing Parameters:")
    print(f"    - Threshold:    {best_thr}")
    print(f"    - Median Filter:{best_med}")
    print(f"    - Min Duration: {best_min_dur}s")
    print(f"    - Clip Gate:    {best_gate}")
    print("=" * 70)

    # Save metrics report and update checkpoint
    metrics = {
        "budget": args.budget,
        "split": args.split,
        "val_clips": len(val_rows),
        "combined_score": float(best_result[0]),
        "event_f1": float(best_result[1]),
        "segment_dice": float(best_result[2]),
        "precision": float(best_result[7]),
        "recall": float(best_result[8]),
        "optimal_postproc": {
            "threshold": float(best_thr),
            "median_filter": int(best_med),
            "min_duration": float(best_min_dur),
            "clip_gate": float(best_gate),
        },
    }

    metrics_file = Path("track1_val_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics written to {metrics_file}")

    ck.update({
        "thr": float(best_thr),
        "med": int(best_med),
        "min_dur": float(best_min_dur),
        "gate": float(best_gate),
        "metrics": metrics,
    })
    torch.save(ck, ckpt_path)
    print(f"[INFO] Best model checkpoint saved to {ckpt_path.resolve()}")


if __name__ == "__main__":
    main()
