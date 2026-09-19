"""Metadata-only analysis script for the Vaani Noise Event dataset (Kaggle-ready).

This script inspects ONLY the metadata columns from all 183 Parquet shards of:
    ARTPARK-IISc/Vaani-Noise-Event-Dataset
without downloading or decoding the ~17.7 GB audio column.

It computes and answers the critical questions for IndoML 2026 Datathon Track 1 & 2:
  1. Per-tier summary: Clip count, duration, and hours across Gold, Silver, and Bronze.
  2. Category distribution: Event counts, hours, % of noise duration, and duration percentiles.
  3. Overlap & nested events: How often do events overlap, and how many are nested?
  4. Union-Oracle Ceiling: If an oracle predicts the exact union mask (Channel 0),
     what Event F1, Segment Dice, and Combined score does it achieve against ground truth?
     (Tests whether the baseline's union-only limitation severely caps Event F1).
  5. Boundary Jitter Sensitivity: How sensitive is Event F1 to 10ms, 20ms, 50ms, 100ms jitter?
  6. District breakdown in Gold: For building a leak-free district-disjoint validation split.

Usage on Kaggle:
  - Add HF_TOKEN to Kaggle Secrets (Add-ons -> Secrets -> label 'HF_TOKEN').
  - Run as a notebook cell or via terminal: `python vaani_meta.py`
  - To test on the first 5 shards: `python vaani_meta.py --num-shards 5`
  - To run all 183 shards: `python vaani_meta.py` (takes ~1-3 minutes)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Hugging Face Token & Filesystem Setup
# ---------------------------------------------------------------------------
def get_hf_token() -> Optional[str]:
    """Retrieve Hugging Face token from Kaggle Secrets, environment, or HF cache."""
    # 1. Try Kaggle secrets
    try:
        from kaggle_secrets import UserSecretsClient
        tok = UserSecretsClient().get_secret("HF_TOKEN")
        if tok:
            print("[INFO] Retrieved HF_TOKEN from Kaggle Secrets.")
            return tok
    except Exception:
        pass

    # 2. Try environment variable
    tok = os.environ.get("HF_TOKEN")
    if tok:
        print("[INFO] Retrieved HF_TOKEN from environment variable.")
        return tok

    # 3. Try Hugging Face Hub token cache
    try:
        from huggingface_hub import HfFolder
        tok = HfFolder.get_token()
        if tok:
            print("[INFO] Retrieved token from huggingface-cli / HfFolder.")
            return tok
    except Exception:
        pass

    print("[WARN] No HF_TOKEN found. Gated dataset access may fail.")
    return None


# ---------------------------------------------------------------------------
# Track 1 Evaluation Metrics (Mirrors Official Scorer)
# ---------------------------------------------------------------------------
def match_events(
    ref_events: List[Tuple[float, float]],
    pred_events: List[Tuple[float, float]],
    tolerance_frac: float = 0.20,
) -> Tuple[int, int, int]:
    """Official Track 1 event matching logic: greedy closest-first."""
    matched_ref = set()
    matched_pred = set()
    candidates = []
    for ri, (r_on, r_off) in enumerate(ref_events):
        tol = max(tolerance_frac * (r_off - r_on), 0.05)
        for pi, (p_on, p_off) in enumerate(pred_events):
            if abs(p_on - r_on) <= tol and abs(p_off - r_off) <= tol:
                candidates.append((abs(p_on - r_on) + abs(p_off - r_off), ri, pi))

    for _, ri, pi in sorted(candidates):
        if ri not in matched_ref and pi not in matched_pred:
            matched_ref.add(ri)
            matched_pred.add(pi)

    tp = len(matched_ref)
    fp = len(pred_events) - tp
    fn = len(ref_events) - tp
    return tp, fp, fn


def event_based_f1(
    ref_dict: Dict[str, List[Tuple[float, float]]],
    pred_dict: Dict[str, List[Tuple[float, float]]],
) -> Dict[str, float]:
    """Micro-averaged event F1 across clips."""
    TP = FP = FN = 0
    for cid, ref_events in ref_dict.items():
        tp, fp, fn = match_events(ref_events, pred_dict.get(cid, []))
        TP += tp
        FP += fp
        FN += fn

    for cid in pred_dict:
        if cid not in ref_dict:
            FP += len(pred_dict[cid])

    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"f1": f1, "precision": prec, "recall": rec, "tp": TP, "fp": FP, "fn": FN}


def events_to_frames(events: List[Tuple[float, float]], max_time: float, frame_len: float = 0.01) -> List[int]:
    """Rasterise (onset, offset) to 10 ms occupancy mask."""
    n = int(max_time / frame_len) + 1
    mask = [0] * n
    for on, off in events:
        s_idx = max(0, int(on / frame_len))
        e_idx = min(int(off / frame_len) + 1, n)
        for i in range(s_idx, e_idx):
            mask[i] = 1
    return mask


def segment_dice(
    ref_dict: Dict[str, List[Tuple[float, float]]],
    pred_dict: Dict[str, List[Tuple[float, float]]],
) -> float:
    """Macro-averaged Segment Dice across clips on 10 ms grid."""
    scores = []
    for cid, ref_events in ref_dict.items():
        pred_events = pred_dict.get(cid, [])
        all_ev = ref_events + pred_events
        if not all_ev:
            scores.append(1.0)
            continue
        max_time = max(off for _, off in all_ev) + 0.5
        rm = events_to_frames(ref_events, max_time)
        pm = events_to_frames(pred_events, max_time)
        inter = sum(r & p for r, p in zip(rm, pm))
        total = sum(rm) + sum(pm)
        scores.append(1.0 if total == 0 else 2.0 * inter / total)
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Interval Geometry & Helpers
# ---------------------------------------------------------------------------
def merge_intervals(spans: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Compute the 1D union of overlapping/adjacent intervals."""
    if not spans:
        return []
    sorted_spans = sorted(spans, key=lambda x: (x[0], x[1]))
    merged = [sorted_spans[0]]
    for cur_s, cur_e in sorted_spans[1:]:
        prev_s, prev_e = merged[-1]
        if cur_s <= prev_e:
            merged[-1] = (prev_s, max(prev_e, cur_e))
        else:
            merged.append((cur_s, cur_e))
    return merged


def parse_spans(raw_spans: Any) -> List[Dict[str, Any]]:
    """Extract valid (start, end, category, tag) from row's timestamp array."""
    out = []
    if raw_spans is None:
        return out

    # Handle if raw_spans is a numpy array or list of dicts
    items = raw_spans
    if hasattr(raw_spans, "tolist"):
        items = raw_spans.tolist()
    elif isinstance(raw_spans, dict):
        items = [raw_spans]

    for s in items:
        if not isinstance(s, dict):
            continue
        try:
            st = float(s.get("start", -1))
            en = float(s.get("end", -1))
            if en > st and st >= 0:
                out.append({
                    "start": st,
                    "end": en,
                    "duration": en - st,
                    "category": str(s.get("category", "")),
                    "tag": str(s.get("tag", "")),
                })
        except (ValueError, TypeError):
            continue
    return sorted(out, key=lambda x: (x["start"], x["end"]))


# ---------------------------------------------------------------------------
# Fast Parquet Metadata Reader
# ---------------------------------------------------------------------------
METADATA_COLS = [
    "imageFileName",
    "state",
    "district",
    "duration",
    "language",
    "annotationQuality",
    "isTranscriptionAvailable",
    "transcript",
    "NoiseCategory",
    "NoiseSubCategoryTimeStamp",
]


def load_all_metadata(
    hf_token: Optional[str],
    num_shards: Optional[int] = None,
    save_cache_path: str = "vaani_metadata.parquet",
) -> pd.DataFrame:
    """Reads metadata columns from HF parquet shards via byte-range requests."""
    # Check if cached locally first
    if os.path.exists(save_cache_path):
        print(f"[INFO] Found local cache: {save_cache_path}. Loading...")
        return pd.read_parquet(save_cache_path)

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=hf_token)
    repo_prefix = "datasets/ARTPARK-IISc/Vaani-Noise-Event-Dataset/data"
    print(f"[INFO] Listing shards in {repo_prefix}...")
    shards = sorted(fs.glob(f"{repo_prefix}/train-*.parquet"))
    print(f"[INFO] Total shards found: {len(shards)}")

    if num_shards is not None and num_shards > 0:
        shards = shards[:num_shards]
        print(f"[INFO] Subsetting to first {len(shards)} shards for quick test.")

    dfs = []
    t0 = time.time()
    for shard_path in tqdm(shards, desc="Reading Parquet Metadata"):
        try:
            table = pq.read_table(
                shard_path,
                filesystem=fs,
                columns=METADATA_COLS,
            )
            dfs.append(table.to_pandas())
        except Exception as e:
            print(f"[WARN] Error reading {shard_path}: {e}")

    if not dfs:
        raise RuntimeError("No parquet shards could be read! Check HF token and permissions.")

    df = pd.concat(dfs, ignore_index=True)
    elapsed = time.time() - t0
    print(f"[INFO] Read {len(df):,} total clips in {elapsed:.1f}s (~{len(df)/max(1, elapsed):.0f} clips/s).")

    # Cache for instant re-runs
    try:
        df.to_parquet(save_cache_path, index=False)
        print(f"[INFO] Saved metadata cache to {save_cache_path} ({os.path.getsize(save_cache_path) / (1024*1024):.1f} MB).")
    except Exception as e:
        print(f"[WARN] Could not cache to {save_cache_path}: {e}")

    return df


# ---------------------------------------------------------------------------
# Metadata Analysis Suite
# ---------------------------------------------------------------------------
def run_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("           DATATHON @ INDOML 2026 — VAANI METADATA AUDIT")
    print("=" * 70)

    results: Dict[str, Any] = {}

    # 1. Tier Overview
    print("\n--- 1. ANNOTATION TIERS OVERVIEW ---")
    tier_summary = {}
    for tier, group in df.groupby("annotationQuality"):
        durations = group["duration"].dropna()
        n_clips = len(group)
        tot_hrs = durations.sum() / 3600.0
        mean_dur = durations.mean()
        med_dur = durations.median()
        min_dur = durations.min()
        max_dur = durations.max()
        tier_summary[tier] = {
            "clips": int(n_clips),
            "hours": float(tot_hrs),
            "mean_dur_sec": float(mean_dur),
            "med_dur_sec": float(med_dur),
            "min_dur_sec": float(min_dur),
            "max_dur_sec": float(max_dur),
        }
        print(f"  * {tier:24s}: {n_clips:>7,} clips | {tot_hrs:>7.2f} hrs | dur: mean {mean_dur:.2f}s, med {med_dur:.2f}s, [{min_dur:.2f}, {max_dur:.2f}]s")
    results["tier_summary"] = tier_summary

    # Parse spans for Gold and Silver
    print("\n--- 2. PARSING TIMESTAMPED EVENTS ---")
    df["parsed_spans"] = df["NoiseSubCategoryTimeStamp"].apply(parse_spans)
    df["n_events"] = df["parsed_spans"].apply(len)

    # Empty clips check
    for tier in ["verified_timestamps", "unverified_timestamps", "no_timestamps"]:
        sub = df[df["annotationQuality"] == tier]
        if len(sub) == 0:
            continue
        empty_cnt = (sub["n_events"] == 0).sum()
        empty_pct = (empty_cnt / len(sub)) * 100
        print(f"  * {tier:24s}: {empty_cnt:,} / {len(sub):,} clips with 0 timestamps ({empty_pct:.2f}%)")

    # 3. Category Distribution in Gold & Silver
    print("\n--- 3. CATEGORY DISTRIBUTION (GOLD DATA) ---")
    gold_df = df[df["annotationQuality"] == "verified_timestamps"].copy()
    silver_df = df[df["annotationQuality"] == "unverified_timestamps"].copy()

    def analyze_categories(tier_df: pd.DataFrame, tier_name: str) -> Dict[str, Any]:
        cat_stats = defaultdict(lambda: {"count": 0, "durations": []})
        total_event_dur = 0.0
        for spans in tier_df["parsed_spans"]:
            for sp in spans:
                c = sp["category"] or "unknown"
                d = sp["duration"]
                cat_stats[c]["count"] += 1
                cat_stats[c]["durations"].append(d)
                total_event_dur += d

        print(f"\n  [{tier_name}] Event Statistics by Category (Total event-time: {total_event_dur/3600.0:.2f} hrs):")
        print(f"  {'Category':<22} | {'Count':>8} | {'Tot (hr)':>8} | {'% Time':>7} | {'Mean (s)':>8} | {'Med (s)':>7} | {'P90 (s)':>7}")
        print("  " + "-" * 78)

        cat_summary = {}
        for c in sorted(cat_stats.keys(), key=lambda k: cat_stats[k]["count"], reverse=True):
            cnt = cat_stats[c]["count"]
            durs = np.array(cat_stats[c]["durations"])
            tot_h = durs.sum() / 3600.0
            pct_t = (durs.sum() / total_event_dur * 100.0) if total_event_dur > 0 else 0
            mean_d = float(np.mean(durs))
            med_d = float(np.median(durs))
            p90_d = float(np.percentile(durs, 90))
            cat_summary[c] = {
                "count": cnt, "hours": tot_h, "pct_time": pct_t,
                "mean_dur": mean_d, "median_dur": med_d, "p90_dur": p90_d,
            }
            print(f"  {c:<22} | {cnt:>8,} | {tot_h:>8.2f} | {pct_t:>6.1f}% | {mean_d:>8.2f} | {med_d:>7.2f} | {p90_d:>7.2f}")
        return cat_summary

    results["gold_category_summary"] = analyze_categories(gold_df, "GOLD (verified_timestamps)")

    # 4. Overlap & Nested Events Analysis
    print("\n--- 4. OVERLAP & NESTED EVENTS AUDIT ---")
    def audit_overlaps(tier_df: pd.DataFrame, tier_name: str) -> Dict[str, Any]:
        clips_with_overlaps = 0
        clips_with_nested = 0
        total_nested_events = 0
        total_events = 0
        total_merged_events = 0
        whole_clip_events = 0

        for _, row in tier_df.iterrows():
            spans = row["parsed_spans"]
            dur = row["duration"]
            if not spans:
                continue

            n = len(spans)
            total_events += n

            # Check whole-clip events (onset <= 0.05s and offset >= dur - 0.05s)
            has_wc = False
            for sp in spans:
                if sp["start"] <= 0.05 and sp["end"] >= (dur - 0.05):
                    whole_clip_events += 1
                    has_wc = True

            # Check overlaps and nesting
            has_overlap = False
            has_nest = False
            for i in range(n):
                for j in range(i + 1, n):
                    s1, e1 = spans[i]["start"], spans[i]["end"]
                    s2, e2 = spans[j]["start"], spans[j]["end"]
                    # Overlap condition
                    if max(s1, s2) < min(e1, e2):
                        has_overlap = True
                        # Nesting condition: one interval is fully inside the other
                        if (s1 <= s2 and e2 <= e1) or (s2 <= s1 and e1 <= e2):
                            has_nest = True
                            total_nested_events += 1

            if has_overlap:
                clips_with_overlaps += 1
            if has_nest:
                clips_with_nested += 1

            # 1D Union reduction
            raw_intervals = [(sp["start"], sp["end"]) for sp in spans]
            merged = merge_intervals(raw_intervals)
            total_merged_events += len(merged)

        n_clips = max(1, len(tier_df))
        stats = {
            "total_clips": n_clips,
            "total_events": total_events,
            "total_merged_events": total_merged_events,
            "events_lost_to_union_merging": total_events - total_merged_events,
            "merge_loss_pct": (total_events - total_merged_events) / max(1, total_events) * 100,
            "clips_with_overlaps": clips_with_overlaps,
            "overlap_clip_pct": (clips_with_overlaps / n_clips) * 100,
            "clips_with_nested": clips_with_nested,
            "nested_clip_pct": (clips_with_nested / n_clips) * 100,
            "total_nested_events": total_nested_events,
            "whole_clip_events": whole_clip_events,
            "whole_clip_event_pct": (whole_clip_events / max(1, total_events)) * 100,
        }

        print(f"  [{tier_name}]:")
        print(f"    Total Events: {total_events:,} -> Merges to {total_merged_events:,} union spans")
        print(f"    Events merged/lost by 1D union: {stats['events_lost_to_union_merging']:,} ({stats['merge_loss_pct']:.2f}%)")
        print(f"    Clips with overlapping events: {clips_with_overlaps:,} / {n_clips:,} ({stats['overlap_clip_pct']:.2f}%)")
        print(f"    Clips with strictly nested events: {clips_with_nested:,} / {n_clips:,} ({stats['nested_clip_pct']:.2f}%)")
        print(f"    Total strictly nested events: {total_nested_events:,}")
        print(f"    Whole-clip events (start~0 to end~dur): {whole_clip_events:,} ({stats['whole_clip_event_pct']:.2f}% of events)")
        return stats

    results["gold_overlaps"] = audit_overlaps(gold_df, "GOLD (verified_timestamps)")
    if len(silver_df) > 0:
        results["silver_overlaps"] = audit_overlaps(silver_df, "SILVER (unverified_timestamps)")

    # 5. Union-Oracle Ceiling Experiment on Gold
    print("\n--- 5. UNION-ORACLE CEILING EXPERIMENT (GOLD DATA) ---")
    print("  Evaluating an oracle that predicts the EXACT 1D Union of all events.")
    print("  This measures the theoretical upper bound for ANY Channel-0 union-only model.")

    ref_dict: Dict[str, List[Tuple[float, float]]] = {}
    oracle_union_dict: Dict[str, List[Tuple[float, float]]] = {}

    for idx, row in gold_df.iterrows():
        cid = f"clip_{idx}"
        raw_intervals = [(sp["start"], sp["end"]) for sp in row["parsed_spans"]]
        ref_dict[cid] = raw_intervals
        oracle_union_dict[cid] = merge_intervals(raw_intervals)

    # Score with official Track 1 metrics
    f1_res = event_based_f1(ref_dict, oracle_union_dict)
    dice_score = segment_dice(ref_dict, oracle_union_dict)
    combined_score = f1_res["f1"] + dice_score

    print("  " + "-" * 50)
    print(f"  Union-Oracle Event F1:        {f1_res['f1']:.4f}")
    print(f"    - Event Precision:          {f1_res['precision']:.4f} (TP={f1_res['tp']:,}, FP={f1_res['fp']:,})")
    print(f"    - Event Recall:             {f1_res['recall']:.4f} (FN={f1_res['fn']:,})")
    print(f"  Union-Oracle Segment Dice:    {dice_score:.4f}")
    print(f"  Union-Oracle Combined Score:  {combined_score:.4f} / 2.00")
    print("  " + "-" * 50)
    print(f"  KEY TAKEAWAY:")
    if f1_res['f1'] < 0.90:
        print(f"  [CRITICAL] A union-only model CANNOT exceed {f1_res['f1']:.2f} F1 on Gold!")
        print(f"  Because nested/overlapping events collapse into 1 span, losing {f1_res['fn']:,} events.")
        print(f"  Multi-class or multi-stream event export is ESSENTIAL to break past the ceiling!")
    else:
        print(f"  [POSITIVE] Union ceiling is {f1_res['f1']:.2f} F1, allowing strong performance with Channel 0.")

    results["union_oracle"] = {
        "event_f1": f1_res["f1"],
        "precision": f1_res["precision"],
        "recall": f1_res["recall"],
        "segment_dice": dice_score,
        "combined_score": combined_score,
        "tp": f1_res["tp"],
        "fp": f1_res["fp"],
        "fn": f1_res["fn"],
    }

    # 6. Boundary Jitter Sensitivity Experiment
    print("\n--- 6. BOUNDARY JITTER CEILING EXPERIMENT (GOLD DATA) ---")
    print("  Evaluating how boundary shifts (e.g. 10ms, 20ms, 50ms) degrade Event F1.")
    jitter_results = {}
    for jitter_ms in [10, 20, 40, 50, 80, 100]:
        shift_sec = jitter_ms / 1000.0
        jitter_dict = {}
        for cid, spans in ref_dict.items():
            perturbed = []
            for s, e in spans:
                # Add random or uniform jitter within [-shift_sec, +shift_sec]
                dur = e - s
                # Keep valid span
                new_s = max(0.0, s + np.random.uniform(-shift_sec, shift_sec))
                new_e = max(new_s + 0.05, e + np.random.uniform(-shift_sec, shift_sec))
                perturbed.append((new_s, new_e))
            jitter_dict[cid] = perturbed

        j_f1 = event_based_f1(ref_dict, jitter_dict)["f1"]
        j_dice = segment_dice(ref_dict, jitter_dict)
        jitter_results[f"{jitter_ms}ms"] = {"f1": j_f1, "dice": j_dice, "combined": j_f1 + j_dice}
        print(f"  * Jitter ±{jitter_ms:>3} ms : Event F1 = {j_f1:.4f} | Dice = {j_dice:.4f} | Combined = {j_f1 + j_dice:.4f}")
    results["jitter_sensitivity"] = jitter_results

    # 7. District Breakdown in Gold (for Train/Val Split Design)
    print("\n--- 7. DISTRICT STRATIFICATION IN GOLD (FOR VALIDATION SPLIT) ---")
    dist_counts = gold_df.groupby("district")["duration"].agg(["count", "sum"]).reset_index()
    dist_counts["hours"] = dist_counts["sum"] / 3600.0
    dist_counts = dist_counts.sort_values(by="count", ascending=False)
    print(f"  Total unique districts in Gold: {len(dist_counts)}")
    print(f"  Top 10 districts by volume in Gold:")
    for _, r in dist_counts.head(10).iterrows():
        print(f"    - {r['district']:<20}: {int(r['count']):>5,} clips | {r['hours']:>5.2f} hrs")

    results["gold_districts"] = dist_counts.to_dict(orient="records")

    # Save summary json
    out_json = "vaani_metadata_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[INFO] Full results written to {out_json}")
    print("=" * 70 + "\n")
    return results


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Vaani Noise Event Dataset Metadata Audit")
    parser.add_argument("--num-shards", type=int, default=None, help="Number of shards to inspect (default: all 183)")
    parser.add_argument("--cache-path", type=str, default="vaani_metadata.parquet", help="Path to cache metadata parquet")
    args = parser.parse_args()

    tok = get_hf_token()
    df = load_all_metadata(hf_token=tok, num_shards=args.num_shards, save_cache_path=args.cache_path)
    run_analysis(df)


if __name__ == "__main__":
    main()
