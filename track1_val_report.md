# IndoML 2026 Track 1 Validation Report — Smoke Run (Fast Budget)

- **Date:** 20 Sep 2026
- **Budget Profile:** `fast` (Smoke/Sanity run: 100 training clips, 68 validation clips)
- **Validation Split:** `district` (Zero speaker leakage)
- **Held-Out Districts:** `['Bhopal', 'Dhar', 'Katni', 'Unakoti']` (68 whole clips)
- **Model Checkpoint:** `t1_wavlm_best.pt` (saved locally in `/kaggle/working/`)

---

## 1. Faithful Whole-Clip Validation Results
Evaluated on complete, uncropped waveforms against raw ground truth spans:

| Metric | Score | Notes |
|---|---|---|
| **Combined Score** | **0.4705** / 2.0000 | Baseline initial smoke checkpoint |
| **Event F1** | **0.0860** | Precision: 0.1176, Recall: 0.0678 |
| **Segment Dice** | **0.3845** | Macro-averaged on 10ms frame grid |

---

## 2. Optimal Post-Processing Parameters (Swept on Whole Clips)
- **Probability Threshold (`thr`):** `0.55`
- **Median Filter Size (`med`):** `11` frames (0.22 s smoothing window)
- **Minimum Duration (`min_dur`):** `0.20 s`
- **Clip Gate (`clip_gate`):** `0.40`

---

## 3. Engineering & Health Check
- **GPU Accelerator:** CUDA active (Nvidia T4 x2)
- **Streaming & Decoding:** Handled 1,617 streamed rows in 36 seconds without downloading raw audio shards.
- **VRAM / Memory:** Peak memory stable, zero CUDA OOMs on 5s mixup batches + 120s uncropped validation clips.
- **Next Phase:** Full scale run (`--budget full`) with uncapped Gold and Silver data to scale from 100 clips to ~39,000 clips.
