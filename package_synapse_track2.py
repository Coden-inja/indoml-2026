"""Package the Track 2 Material Submission for Team "Synapse".

Builds `Synapse.zip` containing a single top-level folder `Synapse/` with:
  - main.py            Inference entry point (exactly 2 args: <input_folder> <output_folder>)
  - requirements.txt
  - README.md          Setup + run + primary contact (mobile & email)
  - model/             t2_masknet_best.pt (the BiGRU MaskNet checkpoint that scored 8.16)
  - predictions.jsonl  Track-1 noise-event predictions used for gate-conditioned enhancement

The pipeline replicates the scoring submission of 20 Sep 2026 (SI-SDR 8.22, dWER -0.06,
Combined 8.16): event-gated BiGRU MaskNet enhancement + SraVaani-1.0 transcription.

Usage:
  python package_synapse_track2.py --ckpt C:\\path\\to\\t2_masknet_best.pt \
      --phone 98XXXXXXXX --email fbyogesh111@gmail.com
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEAM = "Synapse"
PREDICTIONS_FILE = "predictions.jsonl"

MAIN_PY = r'''"""IndoML 2026 Track 2 - Synapse inference entry point.

Reproduces the scored submission of 20 Sep 2026 (SI-SDR 8.22, dWER -0.06, Combined 8.16):
event-gated BiGRU MaskNet enhancement + SraVaani-1.0 transcription.

Usage (exactly two positional arguments):
  python main.py <input_folder> <output_folder>

Outputs:
  - enhanced 16 kHz mono PCM-16 WAV files with the same names as the input audio
  - transcripts.jsonl (Codabench format: {"clip_id": ..., "text": ...})
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

TARGET_SR = 16000
BLEND = 0.80                      # blend weight on enhanced speech inside noise events (match scored run)
PREDICTIONS_FILE = "predictions.jsonl"
CHECKPOINT = "model/t2_masknet_best.pt"
ASR_REPO = "ARTPARK-IISc/SraVaani-1.0"
TRANSCRIPT_NAME = "transcripts.jsonl"


# ---------------------------------------------------------------------------
# BiGRU MaskNet (architecture copied verbatim from train_t2_champion.py)
# ---------------------------------------------------------------------------
class BiGRUMaskNet(nn.Module):
    def __init__(self, n_fft=512, hop_len=160, win_len=512, hidden_size=256, num_layers=3):
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        self.num_bins = n_fft // 2 + 1

        self.register_buffer("window", torch.hann_window(win_len))

        self.in_proj = nn.Linear(self.num_bins, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, self.num_bins),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, T = x.shape
        stft_c = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            win_length=self.win_len,
            window=self.window,
            return_complex=True,
        )
        mag = torch.abs(stft_c)
        phase = torch.angle(stft_c)

        feat = mag.permute(0, 2, 1)
        h = self.in_proj(feat)
        gru_out, _ = self.gru(h)
        mask = self.out_proj(gru_out)
        mask = mask.permute(0, 2, 1)

        est_mag = mag * mask
        est_c = torch.polar(est_mag, phase)

        est_wav = torch.istft(
            est_c,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            win_length=self.win_len,
            window=self.window,
            length=T,
        )
        return est_wav, mask


# ---------------------------------------------------------------------------
# Event mask & gated blend
# ---------------------------------------------------------------------------
def build_event_mask(duration_sec, events, sr=TARGET_SR, win_samples=320):
    total = int(round(duration_sec * sr))
    mask = np.zeros(total, dtype=np.float32)
    for ev in events:
        s = max(0, int(round(ev["onset"] * sr)))
        e = min(total, int(round(ev["offset"] * sr)))
        if e > s:
            mask[s:e] = 1.0
    if win_samples > 1 and mask.max() > 0:
        import scipy.signal
        kernel = np.hanning(win_samples)
        kernel /= kernel.sum()
        mask = scipy.signal.convolve(mask, kernel, mode="same")
        mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
    return mask


def load_events() -> dict:
    p = Path(PREDICTIONS_FILE)
    t1 = {}
    if p.exists():
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    t1[rec["clip_id"]] = rec.get("events", [])
        print(f"[INFO] Loaded Track-1 gating events for {len(t1)} clips.")
    else:
        print("[WARN] predictions.jsonl not found - running full-clip enhancement.")
    return t1


# ---------------------------------------------------------------------------
# SraVaani ASR (same call pattern as the scored run / Codabench audit)
# ---------------------------------------------------------------------------
class SraVaaniTranscriber:
    def __init__(self, hf_token=None, device="cuda"):
        from huggingface_hub import snapshot_download
        from transformers import AutoModel

        token = hf_token or os.environ.get("HF_TOKEN")
        print(f"[INFO] Downloading / loading SraVaani-1.0 from {ASR_REPO}...")
        path = snapshot_download(ASR_REPO, token=token)
        self.asr = AutoModel.from_pretrained(path, trust_remote_code=True).to(device).eval()
        self.device = device
        print(f"[INFO] SraVaani-1.0 ready on {device.upper()}!")

    def transcribe_paths(self, paths):
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


def main():
    if len(sys.argv) < 3:
        print("Usage: python main.py <input_folder> <output_folder>")
        sys.exit(1)

    in_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    ckpt = Path(CHECKPOINT)
    assert ckpt.exists(), f"Checkpoint not found at {ckpt}. Include train/ download t2_masknet_best.pt."

    state = torch.load(ckpt, map_location=device)
    model = BiGRUMaskNet().to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"[INFO] Loaded {CHECKPOINT} (val SI-SDR: {state.get('val_si_sdr', 'n/a')})")

    events_all = load_events()

    files = sorted(p for p in in_dir.rglob("*") if p.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg"))
    print(f"[INFO] Found {len(files)} audio files in {in_dir}")

    enhanced_paths, t0 = [], time.time()
    for p in files:
        cid = p.stem
        raw, sr = sf.read(str(p), dtype="float32")
        if raw.ndim > 1:
            raw = raw.mean(axis=1)
        if sr != TARGET_SR:
            import librosa
            raw = librosa.resample(raw, orig_sr=sr, target_sr=TARGET_SR)
        raw = np.ascontiguousarray(raw, dtype=np.float32).astype(np.float32)

        events = events_all.get(cid, [])
        if events:
            with torch.no_grad():
                inp = torch.from_numpy(raw).unsqueeze(0).to(device)
                enh, _ = model(inp)
                enh_wav = enh[0].cpu().numpy()
            dur = raw.shape[0] / TARGET_SR
            mask = build_event_mask(dur, events)
            if len(mask) < len(raw):
                mask = np.pad(mask, (0, len(raw) - len(mask)))
            mask = mask[: len(raw)]
            final = (1.0 - mask) * raw + mask * (BLEND * enh_wav + (1.0 - BLEND) * raw)
        else:
            final = raw

        pk = np.abs(final).max()
        if pk > 0.99:
            final = final / pk * 0.99

        dst = out_dir / f"{cid}.wav"
        sf.write(str(dst), final.astype(np.float32), TARGET_SR, subtype="PCM_16")
        enhanced_paths.append(dst)

    print(f"[INFO] Enhanced {len(files)} clips in {(time.time() - t0) / 60:.2f} min.")

    print(f"[INFO] Transcribing with {ASR_REPO}...")
    transcriber = SraVaaniTranscriber(device=str(device))
    transcripts = {}
    for i in range(0, len(enhanced_paths), 16):
        batch = enhanced_paths[i: i + 16]
        for p_, txt in zip(batch, transcriber.transcribe_paths([str(x) for x in batch])):
            transcripts[p_.stem] = txt

    with open(out_dir / TRANSCRIPT_NAME, "w", encoding="utf-8") as f:
        for p in files:
            cid = p.stem
            f.write(json.dumps({"clip_id": cid, "text": transcripts.get(cid, "")}, ensure_ascii=False) + "\n")

    print(f"[DONE] Enhanced WAVs + {TRANSCRIPT_NAME} written to {out_dir}")


if __name__ == "__main__":
    main()
'''

REQUIREMENTS_TXT = """torch>=2.0.0
numpy>=1.24.0
soundfile>=0.12.1
librosa>=0.10.0
scipy>=1.10.0
tqdm>=4.65.0
transformers>=4.40.0
huggingface_hub>=0.20.0
"""


def main():
    parser = argparse.ArgumentParser(description="Package Synapse Track 2 material submission ZIP")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to t2_masknet_best.pt (download from Kaggle /kaggle/working/t2_masknet_best.pt)")
    parser.add_argument("--phone", type=str, default="",
                        help="Primary contact mobile number (required in README)")
    parser.add_argument("--email", type=str, default="fbyogesh111@gmail.com",
                        help="Primary contact email")
    parser.add_argument("--author", type=str, default="",
                        help="Primary contact name")
    parser.add_argument("--pred", type=str, default=str(ROOT / "predictions.jsonl"),
                        help="Track-1 predictions.jsonl used for gating")
    args = parser.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        sys.exit(f"[ERROR] Checkpoint not found: {ckpt}. "
                 "Download it from Kaggle (/kaggle/working/t2_masknet_best.pt) first.")

    phone = args.phone.strip()
    email = args.email.strip()
    author = args.author.strip() or "Yogesh Kumar"

    build = ROOT / TEAM
    if build.exists():
        shutil.rmtree(build)
    (build / "model").mkdir(parents=True, exist_ok=True)

    (build / "main.py").write_text(MAIN_PY, encoding="utf-8")
    (build / "requirements.txt").write_text(REQUIREMENTS_TXT, encoding="utf-8")
    shutil.copy2(ckpt, build / "model" / "t2_masknet_best.pt")

    if Path(args.pred).exists():
        shutil.copy2(args.pred, build / PREDICTIONS_FILE)
        print(f"[INFO] Bundled {PREDICTIONS_FILE} ({len(open(args.pred, encoding='utf-8').readlines())} lines)")

    readme = f"""# Synapse — Datathon@IndoML 2026, Track 2 (Noise Event Removal)

## Primary Contact
- **Team Name:** {TEAM}
- **Contact Name:** {author}
- **Mobile:** {phone if phone else "ENTER_YOUR_MOBILE_NUMBER_HERE"}
- **Email:** {email}

## What this reproduces
The submitted score of **20 Sep 2026: Combined 8.16** (SI-SDR 8.22 dB, ΔWER −0.06%).
Pipeline: **BiGRU MaskNet** (STFT → BiGRU → magnitude mask → iSTFT, trained on SI-SDR of the
synthetic validation pairs) with **Track-1 event-gated blending** — speech outside detected
noise events is left 100% untouched, inside noise events the enhanced signal is blended at 0.80 —
followed by **RMS/peak preservation** and transcription with the mandated
`ARTPARK-IISc/SraVaani-1.0` ASR.

No trained SraVaani weights are bundled; the ASR is loaded from Hugging Face at run time as
specified in the competition instructions.

## Files
```
Synapse/
├── main.py              # inference entry point
├── requirements.txt
├── README.md
├── predictions.jsonl    # Track-1 events used for gate-conditioned enhancement
└── model/
    └── t2_masknet_best.pt   # trained BiGRU MaskNet checkpoint
```

## Setup
```bash
pip install -r requirements.txt
```
An internet connection is needed for the first run (downloads SraVaani-1.0 and Hugging Face
models), and a GPU is recommended (enhancement + ASR for ~185 min of audio took ~2.5 h on one T4).

## Run
The script takes exactly two arguments:
```bash
python main.py <input_folder> <output_folder>
```
- `input_folder` — folder containing the test audio files.
- `output_folder` — where results are saved.

Output:
- one enhanced **16 kHz mono PCM-16 WAV** per input clip, **same filename** as input, and
- **`transcripts.jsonl`** (Codabench format: `{{"clip_id": ..., "text": ...}}`).

## Reproducibility notes
- The model checkpoint is the exact best checkpoint used for the score-8 submission.
- Event gating uses the bundled `predictions.jsonl`; clips with no detected events are kept
  byte-identical to the input.
- Deterministic seeds are not needed for inference; no randomness is used at decode time.
"""
    (build / "README.md").write_text(readme, encoding="utf-8")

    zip_path = ROOT / f"{TEAM}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for fp in build.rglob("*"):
            if fp.is_file():
                z.write(fp, f"{TEAM}/{fp.relative_to(build)}")

    shutil.rmtree(build)

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[SUCCESS] {zip_path} ({size_mb:.1f} MB)")
    print("[INFO] Top-level folder inside ZIP: Synapse/")


if __name__ == "__main__":
    main()