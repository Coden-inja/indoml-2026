"""Track 1 Submission Generator.

Loads the trained WavLM SED checkpoint and generates `predictions.jsonl`
and `submission_track1.zip` for Codabench evaluation.

Usage on Kaggle:
  !python predict_track1.py --test-dir /kaggle/working/test_audio/
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
import sys

# Add modular package to python path
pkg_dir = Path(__file__).resolve().parent / "track1_detection" / "versions" / "v-3_indoml2026_track1" / "modular"
sys.path.insert(0, str(pkg_dir))

import torch
import numpy as np
from tqdm.auto import tqdm

from config import CFG, WORK, FRAME_SEC, MIN_SAMPLES
from model import WavLMSED
from inference import t1_posteriors
from evaluate import prob_to_events


def export_track1_tuned(clip_id, wav, model, device, thr, med, min_dur, gate, merge_gap=0.05, sr=CFG["sr"]):
    """Produce the submission record for one clip with precision event boundary resolution."""
    post, clip_p = t1_posteriors(wav, model, device, sr)
    dur = max(len(wav) / sr, MIN_SAMPLES / sr)
    spans = []
    events_iter = [] if clip_p < gate else prob_to_events(post[0], thr=thr, med=med, min_dur=min_dur, merge_gap=merge_gap)
    for on, off in events_iter:
        on, off = float(max(0.0, on)), float(min(dur, off))
        if off - on < min_dur:
            continue
        spans.append([on, off])
    return {
        "clip_id": clip_id,
        "events": [{"onset": round(float(a), 3), "offset": round(float(b), 3)} for a, b in spans],
    }

AUD = (".wav", ".flac", ".mp3", ".ogg")


def read_audio(p: Path, sr: int = CFG["sr"]) -> np.ndarray:
    """Load an audio file to mono float32 at the target sample rate."""
    import soundfile as sf
    import librosa

    w, s = sf.read(str(p), dtype="float32", always_2d=False)
    if w.ndim > 1:
        w = w.mean(axis=1) if w.shape[0] < w.shape[1] else w.mean(axis=0)
    if s != sr:
        w = librosa.resample(w, orig_sr=s, target_sr=sr)
    return np.ascontiguousarray(w, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="IndoML 2026 Track 1 - Inference & Submission Generator")
    parser.add_argument("--test-dir", type=str, default="/kaggle/working/test_audio",
                        help="Path to folder containing test audio files")
    parser.add_argument("--ckpt", type=str, default="t1_wavlm.pt",
                        help="Path to trained checkpoint file")
    parser.add_argument("--output-zip", type=str, default="submission_track1.zip",
                        help="Filename for submission zip")
    parser.add_argument("--thr", type=float, default=0.45,
                        help="Noise detection threshold (default: 0.45)")
    parser.add_argument("--med", type=int, default=7,
                        help="Median filter window size (default: 7)")
    parser.add_argument("--min-dur", type=float, default=0.10,
                        help="Minimum event duration in seconds (default: 0.10s)")
    parser.add_argument("--gate", type=float, default=0.25,
                        help="Clip-level silence gate (default: 0.25)")
    parser.add_argument("--merge-gap", type=float, default=0.05,
                        help="Max gap to merge neighboring events (default: 0.05s)")
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    ckpt_path = Path(args.ckpt)
    out_zip = Path(args.output_zip)
    jsonl_path = Path("predictions.jsonl")

    print("=" * 70)
    print("      INDOML 2026 TRACK 1: INFERENCE & SUBMISSION GENERATOR")
    print("=" * 70)

    if not test_dir.exists():
        # Fallback check in parent, kaggle working, or colab content
        for cand_test in [
            Path("/content/test_audio"),
            Path("/kaggle/working/test_audio"),
            Path("/kaggle/working/input_data"),
            Path("test_audio"),
            Path("input_data"),
        ]:
            if cand_test.exists():
                test_dir = cand_test
                break

    if not test_dir.exists():
        print(f"[ERROR] Test directory not found: {test_dir.resolve()}")
        print("Please download and unzip input_data first.")
        return

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

    # Check candidates
    for cand in [
        Path("/content") / ckpt_path.name,
        Path("/content/t1_wavlm.pt"),
        Path("/content/t1_wavlm.zip"),
        Path("/kaggle/working") / ckpt_path.name,
        Path("/kaggle/working/t1_wavlm.pt"),
        Path("/kaggle/working/t1_wavlm_best.pt"),
        Path("t1_wavlm.pt"),
        Path("t1_wavlm_best.pt"),
        Path(__file__).resolve().parent / "t1_wavlm.pt",
    ]:
        if cand.exists() and cand.is_file():
            if is_valid_torch_file(cand):
                ckpt_path = cand
                break
            else:
                print(f"[WARN] Removing corrupt/stale checkpoint: {cand}")
                cand.unlink()

    # If checkpoint is extracted as a directory (in /kaggle/input or /content)
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
                    # PyTorchFileReader expects the internal paths to start with 't1_wavlm/'
                    shutil.make_archive(str(work_dir / "t1_wavlm"), "zip", root_dir=str(model_dir.parent), base_dir=model_dir.name)
                    if cand_zip.exists():
                        cand_zip.rename(cand_pt)
                    ckpt_path = cand_pt
                    break

    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path.resolve()}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using compute device: {device.upper()}")
    print(f"[INFO] Loading checkpoint: {ckpt_path.resolve()}")

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = WavLMSED().to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    # Retrieve post-processing parameters (CLI overrides checkpoint values)
    thr = args.thr if args.thr is not None else float(ck.get("thr", 0.45))
    med = args.med if args.med is not None else int(ck.get("med", 7))
    min_dur = args.min_dur if args.min_dur is not None else float(ck.get("min_dur", 0.10))
    gate = args.gate if args.gate is not None else float(ck.get("gate", 0.25))
    merge_gap = args.merge_gap if args.merge_gap is not None else float(ck.get("merge_gap", 0.05))
    print(f"[INFO] Using post-processing parameters: thr={thr}, med={med}, min_dur={min_dur}s, gate={gate}, merge_gap={merge_gap}s")

    files = sorted([p for p in test_dir.rglob("*") if p.suffix.lower() in AUD])
    print(f"[INFO] Found {len(files):,} test audio clips in {test_dir.resolve()}")

    if len(files) == 0:
        print("[ERROR] No audio files found in test directory! Check directory structure.")
        return

    print("\n--- RUNNING WHOLE-CLIP INFERENCE (TUNED BOUNDARIES) ---")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for p in tqdm(files, desc="Inference"):
            wav = read_audio(p, sr=CFG["sr"])
            rec = export_track1_tuned(p.stem, wav, model, device, thr=thr, med=med, min_dur=min_dur, gate=gate, merge_gap=merge_gap)
            f.write(json.dumps({"clip_id": p.stem, "events": rec["events"]}, ensure_ascii=False) + "\n")

    # Package into official ZIP format (predictions.jsonl must be at the root of the ZIP)
    print("\n--- PACKAGING SUBMISSION ZIP ---")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(jsonl_path, "predictions.jsonl")

    total_lines = sum(1 for _ in open(jsonl_path, encoding="utf-8"))
    empty_lines = sum(1 for line in open(jsonl_path, encoding="utf-8") if not json.loads(line)["events"])

    print("=" * 70)
    print("                 SUBMISSION READY!")
    print("=" * 70)
    print(f"  Total Predictions: {total_lines:,} clips")
    print(f"  Empty Predictions: {empty_lines:,} clips ({empty_lines/max(1, total_lines)*100:.1f}%)")
    print(f"  Output ZIP File:   {out_zip.resolve()} ({out_zip.stat().st_size / (1024*1024):.2f} MB)")
    print("=" * 70)
    print("You can now download this ZIP from Kaggle and upload to Codabench under 'My Submissions'!")


if __name__ == "__main__":
    main()
