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

from config import CFG, WORK
from model import WavLMSED
from inference import export_track1

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
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    ckpt_path = Path(args.ckpt)
    out_zip = Path(args.output_zip)
    jsonl_path = Path("predictions.jsonl")

    print("=" * 70)
    print("      INDOML 2026 TRACK 1: INFERENCE & SUBMISSION GENERATOR")
    print("=" * 70)

    if not test_dir.exists():
        # Fallback check in parent or kaggle working
        for cand_test in [Path("/kaggle/working/test_audio"), Path("/kaggle/working/input_data"), Path("test_audio"), Path("input_data")]:
            if cand_test.exists():
                test_dir = cand_test
                break

    if not test_dir.exists():
        print(f"[ERROR] Test directory not found: {test_dir.resolve()}")
        print("Please download and unzip input_data first.")
        return

    if not ckpt_path.exists() or ckpt_path.is_dir():
        # Fallback checks in various locations
        for cand in [
            Path("/kaggle/working") / ckpt_path.name,
            Path("/kaggle/working/t1_wavlm.pt"),
            Path("/kaggle/working/t1_wavlm_best.pt"),
            Path("t1_wavlm.pt"),
            Path("t1_wavlm_best.pt"),
            Path(__file__).resolve().parent / "t1_wavlm.pt",
        ]:
            if cand.exists() and cand.is_file():
                ckpt_path = cand
                break

    # If checkpoint is extracted as a directory in /kaggle/input (e.g. from Kaggle dataset upload)
    if not ckpt_path.exists() or ckpt_path.is_dir():
        input_root = Path("/kaggle/input")
        if input_root.exists():
            data_pkls = list(input_root.rglob("data.pkl"))
            if data_pkls:
                model_dir = data_pkls[0].parent
                print(f"[INFO] Detected unzipped checkpoint directory at: {model_dir}")
                print(f"[INFO] Packaging into /kaggle/working/t1_wavlm.pt...")
                import shutil
                cand_zip = Path("/kaggle/working/t1_wavlm.zip")
                cand_pt = Path("/kaggle/working/t1_wavlm.pt")
                if cand_pt.exists():
                    cand_pt.unlink()
                if cand_zip.exists():
                    cand_zip.unlink()
                # PyTorchFileReader expects the internal paths to start with 't1_wavlm/'
                shutil.make_archive("/kaggle/working/t1_wavlm", "zip", root_dir=str(model_dir.parent), base_dir=model_dir.name)
                if cand_zip.exists():
                    cand_zip.rename(cand_pt)
                ckpt_path = cand_pt

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

    # Retrieve optimal swept post-processing parameters
    thr = float(ck.get("thr", 0.50))
    med = int(ck.get("med", 7))
    min_dur = float(ck.get("min_dur", 0.05))
    gate = float(ck.get("gate", 0.0))
    print(f"[INFO] Using post-processing parameters: thr={thr}, med={med}, min_dur={min_dur}s, gate={gate}")

    files = sorted([p for p in test_dir.rglob("*") if p.suffix.lower() in AUD])
    print(f"[INFO] Found {len(files):,} test audio clips in {test_dir.resolve()}")

    if len(files) == 0:
        print("[ERROR] No audio files found in test directory! Check directory structure.")
        return

    print("\n--- RUNNING WHOLE-CLIP INFERENCE ---")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for p in tqdm(files, desc="Inference"):
            wav = read_audio(p, sr=CFG["sr"])
            rec = export_track1(p.stem, wav, model, device, thr, med, min_dur, gate)
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
