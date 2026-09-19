# Datathon@IndoML 2026 — Full Project Context

> Snapshot as of **Sat 19 Sep 2026, ~10:50 PM IST** (Codabench server time 10:46 PM).
> **v2 update: see section 16 at the end.** It supersedes earlier statements in sections 6, 7, 12, 13 and 15 wherever they conflict (Bronze location, label conventions, baseline code, leaderboard, compute).
> Sources: organizer announcement, Codabench pages for both tracks (Overview / Evaluation / Dataset / Terms), and AMA #1 slides (9 Sep 2026).
> Terms say organizers may change rules at any time. Re-check the Codabench pages before any final submission.
> **v2 (Sun 20 Sep 2026, ~00:15 IST): read section 16 at the end first. It supersedes parts of sections 1, 6, 11 and 12** (Bronze location, validation location, the ΔWER convention, model/data policy, baseline findings, leaderboard snapshot).

## 0. How to use this file (for the AI agent)

- Tags used below: **[STATED]** = written on an official page or slide. **[INFERRED]** = a reasonable guess, not confirmed. **[UNKNOWN]** = needs checking against the data, the repo, or the organizers.
- Do not treat [INFERRED] items as facts. Verify them on the downloaded data or the baseline repo first.
- The scorer code in sections 4 and 5 was pasted from the Codabench pages and lost some formatting. Where I reconstructed it, it is marked. **The GitHub baseline repo is the source of truth for scoring code.**
- The user wants to work **slowly and deliberately**, understand why an approach is best, and give input at each step. Do not jump to a full pipeline. Propose, explain trade-offs, then wait.
- The user has **already registered**, and the team is entering **both tracks**. Do not discuss registration.

---

## 1. Clock and stakes

| Item | Value |
|---|---|
| Phase 1 (Half-Marathon) deadline | **22 Sep 2026, 12:00 noon IST** (extended from 20 Sep). About 61 h left at snapshot. |
| Phase 2 (Final) deadline | **17 Oct 2026, 12:00 noon IST** (extended from 15 Oct) |
| Track 2 code ZIP (Google Drive link via Google Form) | Within **2 days after** the Phase 1 deadline. Roughly 24 Sep noon [INFERRED]. Form "shared soon" [STATED]. |
| Leaderboard reset after Phase 1? | **No** [STATED] |

**Note on old dates:** the AMA slides (9 Sep) show 20 Sep and 15 Oct. These are superseded by the 2-day extension, given as compensation for Codabench downtime.

Phase 1 pays: ranks 1–3 get ₹5,000 per team, ranks 4–8 get ₹1,000 per team, per track. The leaderboard carries over into Phase 2.

---

## 2. Structure of the competition

Two separate Codabench competitions, each with its own leaderboard, submission counters and prizes.

| | **Track 1: Noise Event Detection** | **Track 2: Noise Event Removal** |
|---|---|---|
| URL | https://www.codabench.org/competitions/17825/ | https://www.codabench.org/competitions/17835/ |
| Input | Raw audio | Noisy audio + event timestamps |
| Output | `predictions.jsonl` (onset/offset per clip) | Enhanced 16 kHz mono WAVs + `transcripts.jsonl` |
| Metric | Event F1 (±20% tol.) + Segment Dice, max 2.0 | SI-SDR (synthetic only) + 100 × ΔWER (all) |
| Submissions/day, total | 5, 100 | 3, 50 |
| Scoring time limit | 600 s | 3600 s |
| Scorer Docker image | codalab/codalab-legacy:py312 | naguiitkgp/datathon-removal-scoring:latest |
| Code ZIP for Phase 1 | Not required [STATED] | **Required**, see section 9 |
| Participants / submissions (screenshot 8 Sep) | ~148 / 489 | ~80 / 88 |

**How the tracks connect [STATED]:** Track 1's event timeline (onset/offset pairs) is the conditioning input to Track 2, and it tells the denoiser where to suppress noise.
**[UNKNOWN]:** whether the Track 2 test inputs come with organizer-provided (ground-truth) timestamps, or whether we must supply our own from Track 1. The page says "from Track 1 predictions or provided ground truth". Check the files in `input_data`.

**Test set (both tracks):** 11 h withheld = 7 h natural + 4 h synthetic (clean speech with synthetic noise). Test metadata has a `syntheticData` field (true/false). Clip IDs look like `vaani_eval_001`. Test is probably the same 11 h for both tracks [INFERRED].

---

## 3. Task background

- Data comes from **Vaani**: spontaneous, image-prompted Indic speech recorded in real environments (31.2K h, 105 languages, 165 districts). Paper: arxiv.org/abs/2603.28714.
- Motivation: bursty, semantically rich noise events (horns, dogs, crying children, doorbells, appliances, devotional music) degrade Indic ASR. Standard SED datasets (AudioSet, DESED, DCASE) under-represent Indian acoustic contexts.
- Framed under **Responsible AI**: robustness, inclusivity (language-wise and class-wise balance monitored), and transparency (top-5 release code).

---

## 4. Track 1 — Detection

### 4.1 Submission format [STATED]
- ZIP with a **single `predictions.jsonl` at the ZIP root**, no enclosing folder.
- One line per clip: `{"clip_id": "vaani_eval_001", "events": [{"onset": 1.24, "offset": 3.81}, ...]}`. Times in seconds (float).
- `events: []` for no detections. **Every eval clip must appear exactly once** (natural and synthetic).
- **Predictions are class-agnostic:** only time spans are submitted, no category or tag.

### 4.2 Metrics [STATED]
**Combined = Event-F1 + Segment-Dice** (max 2.0). Only the combined score ranks. A natural/synthetic breakdown is shown for transparency and does not affect ranking.

**Event F1 (micro-averaged over all events in all clips).** A prediction is a TP if both boundaries are within tolerance of a reference event.
`tol = max(0.20 × ref_duration, 0.05 s)`, applied to both onset and offset. Matching is greedy, closest first.

**Segment Dice (macro-averaged over clips).** Events are rasterized on a 10 ms grid. Dice = 2·|P∩G| / (|P|+|G|).

Tolerance by reference duration (derived from the formula):

| ref duration | tolerance per boundary |
|---|---|
| ≤ 0.25 s | ±50 ms |
| 0.5 s | ±100 ms |
| 1.0 s | ±200 ms |
| 3.0 s | ±600 ms |

### 4.3 Scorer code (as pasted; indentation preserved)

```python
def match_events(ref_events, pred_events, tolerance_frac=0.20):
    matched_ref, matched_pred = set(), set()
    candidates = []
    for ri, (r_on, r_off) in enumerate(ref_events):
        tol = max(tolerance_frac * (r_off - r_on), 0.05)   # at least 50 ms
        for pi, (p_on, p_off) in enumerate(pred_events):
            if abs(p_on - r_on) <= tol and abs(p_off - r_off) <= tol:
                candidates.append((abs(p_on - r_on) + abs(p_off - r_off), ri, pi))
    for _, ri, pi in sorted(candidates):        # closest first
        if ri not in matched_ref and pi not in matched_pred:
            matched_ref.add(ri); matched_pred.add(pi)
    tp = len(matched_ref)
    return tp, len(pred_events) - tp, len(ref_events) - tp   # tp, fp, fn

def event_based_f1(ref_data, pred_data):
    TP = FP = FN = 0
    for clip_id, ref_events in ref_data.items():
        tp, fp, fn = match_events(ref_events, pred_data.get(clip_id, []))
        TP += tp; FP += fp; FN += fn
    for clip_id in pred_data:                   # extra clips -> false positives
        if clip_id not in ref_data:
            FP += len(pred_data[clip_id])
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    return (2 * precision * recall / (precision + recall)
            if (precision + recall) else 0.0)

def events_to_frames(events, max_time, frame_len=0.01):
    n_frames = int(max_time / frame_len) + 1
    mask = [0] * n_frames
    for on, off in events:
        for i in range(int(on / frame_len), min(int(off / frame_len) + 1, n_frames)):
            mask[i] = 1
    return mask

def segment_dice(ref_data, pred_data):
    scores = []
    for clip_id, ref_events in ref_data.items():
        pred_events = pred_data.get(clip_id, [])
        all_events = ref_events + pred_events
        if not all_events:
            scores.append(1.0); continue
        max_time = max(off for _, off in all_events) + 0.5   # pad 0.5 s
        ref_mask  = events_to_frames(ref_events,  max_time)
        pred_mask = events_to_frames(pred_events, max_time)
        inter = sum(r & p for r, p in zip(ref_mask, pred_mask))
        total = sum(ref_mask) + sum(pred_mask)
        scores.append(1.0 if total == 0 else 2.0 * inter / total)
    return sum(scores) / len(scores) if scores else 0.0
```

Submission packaging (as pasted): write `predictions.jsonl` with `json.dumps({"clip_id":..., "events":[{"onset":..,"offset":..}]}, ensure_ascii=False)`, then `zipfile.ZipFile("submission.zip","w",ZIP_DEFLATED).write("predictions.jsonl","predictions.jsonl")`.

### 4.4 Consequences of the scorer (derived from the code, [STATED] behavior)
1. **Short events dominate F1.** F1 is micro-averaged over events, and non-speech-human events (~35% of all events, mean 0.4 s) get only ~±80 ms tolerance or less. Boundary precision on short events likely decides most of the F1 half.
2. **Empty clips are all-or-nothing for Dice.** Ref empty and pred empty gives 1.0. Ref empty and pred non-empty gives 0.0. Ref non-empty and pred empty gives 0.0. A clip-level "any noise?" decision matters if empty clips exist. **[UNKNOWN]:** the fraction of empty clips.
3. **Fragmenting or merging hurts F1 but barely affects Dice.** Splitting one reference event into two, or merging two into one, gives FPs and FNs. This needs matching the annotators' conventions (for example, one event for a dog-bark run, or many).
4. **Dice uses a per-clip grid ending at last event + 0.5 s**, so true clip length does not enter the score.
5. **Dice unions masks**, but F1 matches events one to one. Overlapping reference events (horn during breathing) can be an issue for a merged blob.
6. **Boundary error costs differ.** Dice has no tolerance but is smooth. F1 is a hard threshold, and short events are the most sensitive. Post-processing (threshold, smoothing, min duration, gap-merge) should be tuned on **F1 + Dice jointly**.
7. **Extra or missing clips:** clips missing from predictions count as all-FN. Extra clip IDs count as FP.

---

## 5. Track 2 — Removal

### 5.1 Submission format [STATED]
ZIP with, **all at the root (no subfolder)**:
- One **WAV per eval clip**: PCM 16-bit signed, **16,000 Hz, mono**, **same duration as original**, filename = clip ID exactly (`vaani_eval_001.wav`). Missing files get a **−50 dB** penalty.
- A single **`transcripts.jsonl`**: `{"clip_id": "vaani_eval_001", "text": "..."}` per line. `clip_id` = WAV filename without `.wav`. Missing or empty transcript counts as WER = 1.0 for that clip.
- Transcripts must come from the **mandated ASR `ARTPARK-IISc/SraVaani-1.0` (frozen, unchanged)**, run on **our enhanced audio**.

### 5.2 Metrics [STATED]
**Combined = SI-SDR(synthetic subset) + 100 × ΔWER(full 11 h)**

- **SI-SDR** (dB, mean over the 4 h synthetic clips only, which have clean references). Signals are truncated to the shorter length. The value is clipped to [−100, 100]. Typical range −5 to 30 dB.
- **ΔWER = WER_noisy − WER_enhanced**, computed over all 11 h. `WER_noisy` is a fixed organizer baseline (mandated ASR on the original noisy clips, identical for everyone). `WER_enhanced` uses our `transcripts.jsonl` against private ground truth. WER is pooled via `jiwer.wer` over lists. Clips with empty GT are dropped.
- ΔWER is a fraction in the formula (×100 to percentage points). The leaderboard column shows it as a percentage (example "−1.17"). **[UNKNOWN]** whether "100 × ΔWER" uses the fraction or the already-percent value. Confirm from the first leaderboard row.
- **PESQ** is also mentioned in the AMA slides "for top-5", but it is **not** in the Codabench combined score. [UNKNOWN] how it is used.

### 5.3 Anti-cheat ASR audit [STATED]
The scorer re-runs the mandated ASR on a **hidden random subset of our submitted WAVs** and compares the output with our `transcripts.jsonl`. Too many mismatches gives **Audit = FAIL** (0), and the submission is manually reviewed and may be disqualified before prizes. Audit values: 1 = PASS, 0 = FAIL, −1 = SKIPPED (never blocks scoring). Using a different ASR or hand-editing transcripts fails the audit.
**Practical consequence [INFERRED]:** always transcribe the **final 16-bit WAV files as saved on disk** (not float audio in memory), with the exact model and decoding settings the scorer would use.

### 5.4 Scorer code (reconstructed; the pasted regex and indentation were damaged)

```python
import re, unicodedata
import numpy as np
from jiwer import wer

# NOTE: pasted as r"</?[^<>]>|[[^[]]]" which lost characters. Best guess of the intent:
TAG_RE = re.compile(r"</?[^<>]+>|\[[^\[\]]+\]")   # drop <...> and [...] tags

def normalize(text):
    s = TAG_RE.sub(" ", text or "")
    s = "".join(" " if unicodedata.category(c).startswith("P") else c for c in s)  # punctuation -> space
    return " ".join(s.lower().split())                                             # lowercase + collapse

def si_sdr(reference, enhanced):
    ref = np.asarray(reference, dtype=np.float64)
    enh = np.asarray(enhanced,  dtype=np.float64)
    n = min(len(ref), len(enh))
    ref, enh = ref[:n], enh[:n]
    scale    = np.dot(enh, ref) / np.dot(ref, ref)
    s_target = scale * ref
    e_noise  = enh - s_target
    value = 10.0 * np.log10(np.dot(s_target, s_target) / np.dot(e_noise, e_noise))
    return float(np.clip(value, -100.0, 100.0))

def delta_wer(gt, noisy_asr, submitted, clip_ids):
    refs, noisy, enh = [], [], []
    for cid in clip_ids:
        g = normalize(gt[cid])
        if not g:                                   # drop clips whose GT is empty
            continue
        refs.append(g)
        noisy.append(normalize(noisy_asr.get(cid, "")) or "@")   # "@" matches nothing
        enh.append(normalize(submitted.get(cid, "")) or "@")
    return wer(refs, noisy) - wer(refs, enh)        # fraction; x100 = percent
```

Packaging (as pasted): write `transcripts.jsonl`, then a ZIP with `transcripts.jsonl` and every `*.wav` written at the root via `z.write(path, name)`.

### 5.5 Consequences and observations (derived, partly [INFERRED])
1. **Identity baseline:** submitting the noisy audio unchanged gives ΔWER = 0 (same ASR, same audio) and SI-SDR equal to the noisy input's SI-SDR. Any enhancer must beat this. Denoisers often add artifacts that **raise** WER, so ΔWER can go negative.
2. **The two terms measure different things.** SI-SDR rewards waveform fidelity on synthetic clips, while ΔWER rewards ASR-friendliness on all clips (natural included). Both count.
3. **ASR-aware training is a natural idea.** The ASR is frozen and public, so enhancement can be optimized for it. Anything used must stay audit-safe: the final WAV must reproduce the submitted transcript.
4. **Duration and format must match exactly.** A wrong length, sample rate or missing file is penalized or breaks the audit.
5. **Timestamps as conditioning.** Slides describe a DeepFilterNet-style magnitude-mask denoiser conditioned on the event timeline. See section 7.

---

## 6. Data

### 6.1 Training tiers [STATED]
| Tier | Size | Content |
|---|---|---|
| 🥇 **Gold** | 11,111 segments, **21.85 h** | Timestamps verified: internal team independently re-timestamped, then a 10% random audit passed |
| 🥈 **Silver** | 61,645 segments, **100.32 h** | External annotators' timestamps, passed a sanity check, agreement **not** verified |
| 🥉 **Bronze** | ~30 h | **No timestamps**. Only noise tags inside the transcript |

Combined Gold + Silver: 72,756 segments, **122.17 h**, **106,892 noise events**, 58 languages, 30 states / 162 districts, 38,541 speakers.
Bronze ≈ 30 h would bring the total to ~150 h [INFERRED]. **[UNKNOWN]** where Bronze is hosted. Check the HF card and https://indoml.in/datathon/#dataset.

### 6.2 Annotation format examples [STATED]
```json
// Gold (Silver is identical minus the Verification_status entry)
[ {"category": "vehicle_traffic", "tag": "<horn>", "start": "2.714", "end": "3.761"},
  {"category": "human_non_speech", "tag": "[breathing]", "start": "4.938", "end": "5.410"},
  {"Verification_status": "Verified"} ]
```
Bronze (transcript only, no times):
`<noise> सजावट के <horn> </horn> लिए यहाँ एक गुलाब भी <horn> </horn> लगाया गया है। </noise>`

Annotation-tool screenshots show other tags: `<static noise>` (spanning a whole clip, 0–6.73 s), `<talking>`, `[breathing]`, `[inhaling]`, `<PAUSE>`, `{building}` (curly braces around a word, possibly English/loanword), `[autoGeneratedTag]`.
**[UNKNOWN]:** whether whole-clip tags such as `<static noise>` appear in the released labels or the evaluation references. If they do, they strongly affect Dice. **First thing to check on the data.**

### 6.3 Seven-class taxonomy [STATED]
| Category | % segs | Events | Mean duration |
|---|---|---|---|
| Non-speech human | 37.8% | 37,739 | 0.4 s |
| Animal | 31.3% | 24,601 | 3.3 s |
| Vehicle / traffic | 24.9% | 20,603 | 2.2 s |
| Baby / child | 16.1% | 12,376 | 3.1 s |
| Singing / music | 9.5% | 6,978 | 5.0 s |
| Phone / signal / alarm | 4.9% | 3,683 | 1.8 s |
| Appliance / machine | 1.3% | 912 | 6.1 s |

Classes can co-occur ("% segs" need not sum to 100%). Class imbalance is strong. Events may overlap speech.

### 6.4 Splits [STATED, with inconsistencies]
| Split | Total | Natural | Synthetic | Availability |
|---|---|---|---|---|
| Train | "20 h" | 20 h | none | "open-sourced" |
| Validation | 4 h | 3 h | 1 h | "planned Hugging Face release" (the Overview says it is in Codabench Files) |
| Test | 11 h | 7 h | 4 h | withheld |

The "Train 20 h" row seems to describe only Gold, while the tier table lists ~150 h. **[UNKNOWN]:** what is actually downloadable. Reconcile after download.
**There is no synthetic training data.** The synthetic test subset (~36% of test hours, and the only part with clean references for SI-SDR) needs our own simulation (clean speech + noise mixtures). The 1 h synthetic validation portion is the only labelled synthetic sample.

### 6.5 Where to get things
- Training data: https://huggingface.co/datasets/ARTPARK-IISc/Vaani-Noise-Event-Dataset (audio + metadata only; images not needed; download code on the card).
- Test audio (`input_data`) and validation set: Codabench competition page, **Get Started → Files** (login required).
- Full dataset info: https://indoml.in/datathon/#dataset

---

## 7. Baselines and organizer hints (from the AMA slides)

- **Track 1 baseline [STATED, quiz answer]:** WavLM-base+ encoder (its 50 Hz frames match the 20 ms label grid) → BiGRU → sigmoid heads, per-frame **BCE over the 7 categories**. So baseline outputs are quantized to 20 ms.
- **Track 2 baseline [STATED]:** DeepFilterNet-style noise-suppression model that predicts a **magnitude mask on the STFT**, then iSTFT back to audio. Training objective: denoising loss, **span-weighted** using the event timestamps. A "SraVaani FastConformer" is described as Track 2's (presumably the ASR side).
- Slides on what timestamps enable: error attribution (which event caused which ASR error), in-span versus out-of-span evaluation, and span-supervised training (weight the loss where the noise is).
- Baselines for both tracks: https://github.com/nagshubhadip/noise-event-detection-and-removal (**[UNKNOWN]** contents; should include scorer code, baselines, maybe the dataset loader). Need to be cloned and read.

---

## 8. Rules and terms [STATED]

- **Indian participants only.** One account per participant/team. Registration approved and closed.
- **External data allowed, must be disclosed** in the system description. **Pre-trained models allowed, must be documented.**
- Dataset only for this competition. No redistribution or commercial use.
- Submissions must be original work with no malicious content.
- **"The final submission on the leaderboard at the deadline is used for ranking."** **[UNKNOWN]** whether this is the last or the best submission. Until clarified, make the last submission the best one.
- **Top-5 per track** must release training and inference code under a permissive licence (**MIT / Apache 2.0**), submit a **≤4-page system description**, and document all pre-trained dependencies. Any model we adopt must have a licence compatible with release.
- Organizers can verify submissions and disqualify violators. Track 2 has the ASR audit.
- **[UNKNOWN]** whether using the unlabelled test audio for self-training or semi-supervision is allowed. The novelty criteria explicitly reward "semi-supervised approaches exploiting unlabelled Vaani audio". Ask the organizers about test audio specifically.

---

## 9. Track 2 Phase-1 code submission requirement [STATED]

For the **final Phase 1 submission**, Track 2 teams must also submit model and inference code for verification, uploaded as a ZIP to Google Drive, with the link sent via the Google Form (shared separately) within 2 days after the Phase 1 deadline.

ZIP must contain:
- Model files (checkpoint and everything to load and run it).
- `requirements.txt` or `environment.yml`.
- **`main.py`** (name it exactly this) taking **two arguments: `input_folder` and `output_folder`**. It writes enhanced audio with the **same filenames as the input**, plus a **JSON transcript in Codabench format**.
- **README** with setup and run instructions and the **primary contact's mobile number and email**.
- **Naming:** ZIP = `Team_name.zip`, containing a single top-level folder `Team_name/`.
- Do **not** include the SraVaani ASR model. Just the script that calls `ARTPARK-IISc/SraVaani-1.0` from Hugging Face. Keep the ZIP lightweight.

---

## 10. Prizes and ranking

- Total pool ₹2,00,000 (₹1,00,000 per track) = Half-Marathon ₹40K + Final ₹1.5L + Spotlight ₹10K. Plus invitations to present at IndoML 2026 and a Novelty Award.
- **Phase 1** (per track): ranks 1–3 ₹5,000 each (₹15,000), ranks 4–8 ₹1,000 each (₹5,000).
- **Final** (per track): 1st ₹35,000, 2nd ₹15,000, 3rd ₹10,000, 4th–8th ₹3,000 each (₹15,000). Total ₹75,000 per track.
- **Track 2 only:** extra **Spotlight Solution Award ₹10,000** for exceptional solution(s) (organizers call Track 2 "the harder problem").
- **Final ranking = Leaderboard 80% + Novelty 15% + Report 5%.** An expert panel applies a novelty factor to the **top-5** metric entries.
- **Novelty rewards:** new architectures, novel training regimes, principled event-conditioning, semi-supervised use of unlabelled Vaani audio, creative use of the multi-tier annotations.
- Participation at AMA time (9 Sep): 275+ registered teams, 45 submitted in Track 1, 15 in Track 2.

---

## 11. Links and resources

| Resource | Status |
|---|---|
| Track 1 Codabench: https://www.codabench.org/competitions/17825/ | known |
| Track 2 Codabench: https://www.codabench.org/competitions/17835/ | known |
| HF dataset: https://huggingface.co/datasets/ARTPARK-IISc/Vaani-Noise-Event-Dataset | known |
| Baseline repo: https://github.com/nagshubhadip/noise-event-detection-and-removal | known, not yet read |
| Competition website: https://indoml.in/datathon (dataset info: `/#dataset`) | known |
| Vaani paper: https://arxiv.org/abs/2603.28714 | known, not yet read |
| Mandated ASR: `ARTPARK-IISc/SraVaani-1.0` (Hugging Face) | known model ID |
| WavLM-base+: `microsoft/wavlm-base-plus` on HF | ID inferred from "WavLM-base+", confirm in the repo |
| AMA #1 recording (YouTube) and slides PDF | linked on Codabench, URLs not captured |
| Google Form for the Track 2 code ZIP | not shared yet |
| Discord (organizers post updates) | link not captured |
| Contact: datathon@indoml.in | known |

Organizers: Dr. Subhajit Datta (Heritage Institute of Technology), Dr. Mahesh Mohan (IIT Kharagpur), Dr. Prasanta Kumar Ghosh (IISc Bangalore), Dr. Debopriyo Banerjee (Inception – G42), Shubhadip Nag (Walmart; technical host, Codabench organizer "nagu_iitkgp"). Technical volunteers: Shivay Vadhera (IIT Bombay), Nihar Desai, Pavan Kumar J, Sujith P (ARTPARK, IISc).

---

## 12. Open questions and discrepancies (resolve early)

1. **Bronze data location**, and whether Bronze is downloadable at all.
2. **Whole-clip tags** (`<static noise>` etc.) in labels and evaluation references?
3. **Fraction of empty clips** (no noise) in train, validation, and probably test.
4. **Validation set location** (HF versus Codabench Files) and how well it mirrors the test mix (7 h natural : 4 h synthetic).
5. **Track 2 test timestamps:** provided by organizers, or must we supply them (from our own Track 1 detector)?
6. **"100 × ΔWER"** fraction versus percent (check the first leaderboard rows).
7. **"Final submission at the deadline"** means last or best?
8. **Test-audio self-training** allowed?
9. **What the "±20% tolerance" really rewards:** measure achievable F1 on Gold by comparing Gold to Silver on the same segments, if overlapping segments exist [idea, not confirmed].
10. **PESQ role** for the top-5.
11. **Class-conditional or class-aware training** helps when the submission is class-agnostic? Test empirically.

---

## 13. Approach families considered (nothing decided)

### Track 1
1. **Frame-level SED with a small CRNN/CNN from scratch**, plus thresholding and smoothing. The classic DCASE-style baseline.
2. **Fine-tune a pretrained encoder** (WavLM-base+ per organizer baseline; also candidates like BEATs/PANNs or speech-oriented encoders) with a frame-level head. Stronger with limited clean labels. Domain fit for Indian conditions must be checked.
3. **Noisy-label learning across tiers:** tier-weighted loss, pretrain on Silver and fine-tune on Gold, pseudo-labelling or mean-teacher, use Gold to correct Silver. Bronze may give coarse position from tag order via forced alignment (an idea, unverified).
4. **Boundary-focused output:** predicting onset/offset directly or with boundary heads, aimed at the F1 tolerance.
5. **Metric-aware post-processing and ensembling:** thresholds, hysteresis, min duration, gap-merge, tuned on validation for F1 + Dice jointly. Clip-level "empty clip" gate.
6. **Synthetic augmentation:** mixtures of clean speech with noise events to cover the synthetic third of the test.
7. **Multi-class heads (7 categories) flattened to class-agnostic spans** for submission.

### Track 2
1. **Identity baseline** (upload noisy audio) as a floor.
2. **DeepFilterNet-style mask** conditioned on the event timeline, trained with a span-weighted loss (the organizer baseline design).
3. **ASR-aware or perceptual losses**, or joint optimization against the frozen SraVaani ASR. Must stay audit-safe.
4. **Bypass when no events** (only touch flagged spans), to protect speech and ΔWER.
5. **Synthetic training pairs** (clean + noise event), since natural clips have no clean reference.
6. **Conditioning on Track 1 detections** (our own) if organizer timestamps are not provided at test time.

### Shared / strategy (tentative)
- Phase 1 is a checkpoint, and Phase 2 has ~4 more weeks with the leaderboard not reset. A sound simple submission for each track first, then the stronger method for Phase 2.
- Novelty is 15% of the final ranking and only applies to the top-5, so a distinct, well-motivated training regime (for example tier-aware training, or timestamp-conditioned enhancement) is worth planning early.

---

## 14. Working plan (tentative, to be confirmed with the user step by step)

1. **Collect data and rules:** HF training data (`snapshot_download`, skip images), Codabench `input_data` and validation (manual, logged in), `git clone` the baseline repo, read the Vaani paper. No scraper needed except an optional Playwright snapshot of the Codabench pages, since rules may change.
2. **Explore data (script):** per tier, event duration histogram, events per clip, share of empty clips, class counts, gaps between same-class events, presence of whole-clip tags, Gold versus Silver agreement where possible, natural versus synthetic stats on validation.
3. **Reproduce the scorer locally** (section 4.3 and 5.4, checked against the repo) and score the organizer baseline on validation.
4. **Minimal valid submissions first** for both tracks (verify formats against the scorer): Track 1 baseline `predictions.jsonl`; Track 2 identity or baseline WAVs + transcripts.
5. **Track 1 modelling** per section 13, then Track 2, then a joint pipeline.
6. **Before Phase 1 ends (Track 2):** make sure `main.py` and the ZIP layout of section 9 work on a fresh folder of audio.
7. **Keep an experiment log** (config, val F1, val Dice, submission scores) and an external-data / pretrained-model list for the system description.

---

## 15. Environment notes

- Plan: develop in **Google Antigravity** (agentic IDE) with **Google Colab** for GPU training. Keep this file in the repo root and point the agent to it.
- Approximate data volume: 150 h at 16 kHz mono is roughly 17 GB as uncompressed WAV, so plan Drive/Colab storage accordingly.
- The Hugging Face dataset may need an HF token (it may be gated). Codabench files need a manual login. A sandboxed assistant cannot reach Hugging Face or log into Codabench, so downloads run on the user's machine or Colab.
- Never paste passwords or tokens into chats or repos.
- Keep Track 2 `main.py` reproducible: fixed seeds, pinned `requirements.txt`, and inference that runs end to end from `input_folder` to `output_folder`.

---

## 16. UPDATE v2 (20 Sep 2026) — supersedes earlier sections where they conflict

### 16.1 Sources read since v1
HF dataset card + ~100 viewer rows; organizer baseline `track1_detection/versions/v-3_indoml2026_track1` (config, data, model, losses, train, evaluate, inference, README); noise dataset paper **arXiv 2609.02474** (full text); Vaani corpus paper 2603.28714 (general corpus, no noise methods); Track 1 leaderboard (page 1 of 2, 50 of 70 teams).
Not yet seen: Track 2 baseline code, the Codabench Files (input_data, validation), full-dataset statistics.

### 16.2 Corrections to earlier statements
1. **Bronze IS on Hugging Face.** `ARTPARK-IISc/Vaani-Noise-Event-Dataset` is one `train` split of 90,637 rows (~154.6 h, 17.7 GB, gated, CC-BY-4.0), told apart by `annotationQuality`: verified_timestamps 11,111 (~21.8 h), unverified_timestamps 61,642 (~100.3 h), no_timestamps 17,884 (~32.4 h).
2. Fields: audio, imageFileName, state, district, duration, language, annotationQuality, isTranscriptionAvailable, transcript, NoiseCategory (list), NoiseSubCategoryTimeStamp (list of {category, tag, start, end}; start/end are STRINGS). **No speaker ID and no unique clip ID** (imageFileName repeats). The Codabench "20 h train" row means Gold only.
3. **The noise dataset's own paper is arXiv 2609.02474** (2 Sep 2026, Pavan Kumar J et al.). 2603.28714 is the general Vaani paper.
4. **A further 10 h of verified data is held out as a speaker-disjoint eval set.** Inference: this is the 3 h natural validation + 7 h natural test.
5. **Gold and Silver are disjoint clip sets** (paper Sec. IV step 3: the sanity-checked pool is split into ~100 h released as-is and a >=20 h candidate subset). A Gold-vs-Silver paired agreement study is NOT possible from the release. The internal team re-timestamped the Gold candidates **with the freelancers' timestamps as reference**, so Gold boundary conventions are probably close to freelancer conventions. Audit rule: if even one event in the 10% sample disagrees, redo the batch.
6. **Almost no clean clips.** The paper's release-quality subset requires NoiseCategory NOT NULL: all 72,756 segments have >=1 category and 72,746 have >=1 timestamped event (10 without). Test natural clips likely follow the same filter. **The baseline's clip gate is probably useless** (a gate can only lose points if nearly every clip has an event). To verify on the official validation set.
7. Transcript wrapper tags `<noise>`, `<static noise>`, `<talking>`, `<people talking>`, `<wind noise>`, `<thudding>` appear in transcripts but never as timestamped events (label space = 7 categories only). Curly braces `{...}` in transcripts hold corrected spellings (Vaani paper App. E.1), not noise; strip when parsing Bronze transcripts. `--` marks an incomplete utterance; `<PAUSE>` marks pauses > 0.5 s.
8. Timestamped events can be **whole-clip** (e.g. `<insect noise>`, `<engine sound>`, `<vehicle noise>`, `<bird squawking>` from 0.000 to the end), seen in Silver from Washim, Purulia, Pathankot and also occasionally in Gold (`<vehicle horn>` 0.000-5.819). Looks like an annotator/district convention. Whole-clip events often contain shorter events (breathing) nested inside.
9. **Retracted:** my earlier inference that leaderboard F1 0.78 implies mostly long, easy events. 35% of all events are brief non-speech human sounds (mean 0.42 s, tolerance ~84 ms), so leaders must also match many short events.

### 16.3 Dataset statistics (paper Tables II-III; release-quality subset = Gold + Silver)
- 72,756 segments, 122.17 h, 38,541 speakers, 58 languages, 30 states / 162 districts. Segment duration min / mean / max = 0.79 / 6.05 / 23.49 s. 106,892 timestamped events (about 1.47 per clip).
- Single category: 55,330 segments; two or more categories: 17,426 (24%). Events may overlap each other and speech.
- Hindi 83.9 h (47,080 segments, about 69% of hours), Telugu 16.7 h, Bengali 12.9 h, Marathi 5.3 h. Top states: Bihar 24.0 h, Andhra Pradesh 14.3 h, Uttar Pradesh 12.9 h, West Bengal 12.3 h, Maharashtra 11.9 h.

| Category | Segments | Events | Event hours | Mean event |
|---|---|---|---|---|
| Non-speech human | 27,533 (37.8%) | 37,739 (35% of events) | 4.5 | 0.42 s |
| Animal | 22,746 | 24,601 | 22.4 | 3.3 s |
| Vehicle / traffic | 18,150 | 20,603 | 12.5 | 2.2 s |
| Baby / child | 11,708 | 12,376 | 10.5 | 3.1 s |
| Singing / music | 6,879 | 6,978 | 9.7 | 5.0 s |
| Phone / signal / alarm | 3,546 | 3,683 | 1.8 | 1.8 s |
| Appliance / machine | 906 | 912 | 1.5 | 6.1 s |

Summed event durations = 62.9 h of 122.17 h audio (about 51%, an upper bound on noise coverage since overlaps double count). A whole-clip prediction at noise fraction f scores Dice = 2f/(1+f), about 0.57-0.67 for f = 0.4-0.5. This matches the ~0.6 Dice at the bottom of the leaderboard [INFERRED, verify with a trivial-predictor experiment].
Underlying database has flags `hasIssue`, `evalSet`, `syntheticData` (not in the HF release). The organizers' 4 h synthetic test clips (clean speech + synthetic noise) are unseen. **How they were mixed (noise source, SNR, placement, boundary sharpness) is UNKNOWN; inspect the 1 h synthetic validation portion.**

### 16.4 Baseline v-3 (organizer code) — what it does
- WavLM-base+ (last 2 of 12 layers dropped, last layer only used) -> 2-layer BiGRU (256/dir) -> frame head + attention head (MIL clip prob), 8 channels (any-noise + 7 categories). 5 s random-crop windows, 20 ms output frames (10 ms labels max-pooled by 2).
- Loss = l_strong (positive-weighted BCE x4 on frames, Gold+Silver only, Silver weight 0.4) + l_weak (clip-tag BCE, all tiers incl. Bronze) + ramped mean-teacher consistency (EMA 0.999, max weight 2.0). Mixup (union labels, p 0.5), time-mask augmentation. Two stages: encoder frozen (3 or 6 epochs) then unfrozen (enc lr 5e-5, head lr 3e-4, cosine; 8 or 16 epochs).
- Inference: whole clip in one pass (windowed above 120 s); ONLY channel 0 is exported; threshold + median filter + gap merge (0.10 s) + min duration; post-processing swept on validation (thr, median, min_dur, clip gate). Sub-50 ms events dropped.
- Budget switch: "fast" (~3 h; default trains on only 100 clips, a smoke test) versus "full" (~7 h; caps Silver at 20k of 61.6k, Bronze at 8k of 17.9k, Gold uncapped). All audio held in RAM as int16 (~8 GB for the full config).
- Scorer functions are copied verbatim from the Codabench page.

### 16.5 Suspected weaknesses of the baseline (HYPOTHESES to test, none verified)
1. Only the union channel is exported, so nested or overlapping events collapse into one run. Cost shows up in event F1.
2. Local validation reference is rebuilt by thresholding the union label mask at 20 ms (`cache_posteriors`), so nested events are merged in the reference too. **Local F1 is not comparable with the leaderboard.**
3. Validation = random 10% of Gold clips (same speakers and sessions as train, and no speaker ID to split on) and only the first 5 s of each clip (mean clip 6.05 s, max 23.5 s). Post-processing settings are tuned on this, so expect optimism.
4. `load_stream` does not shuffle: tier caps take the first N rows, likely geographically clumped.
5. Bronze weight 0.2 (`QW`) is never applied to `l_weak`; Silver and Bronze both get full weight there. Mixup can mix strong and weak clips without adjusting `strong` or `w`.
6. No boundary-aware loss; frame loss averages over 8 channels so the exported channel gets 1/8 of the weight.
7. Checkpoints are chosen on the flawed validation score with fixed default post-processing.
8. The clip gate is probably useless (16.2 item 6).
**Better validation design to build:** official Codabench validation (3 h natural + 1 h synthetic, probably from the speaker-disjoint verified held-out set) for final selection and post-processing, plus a district-disjoint Gold hold-out during development; whole clips; raw reference events (not rasterized). CHECK that the Codabench validation file actually contains reference timestamps.

### 16.6 Leaderboard snapshot (Track 1, server time 19 Sep 22:46 IST; 70 teams, 50 shown)
| Rank | Combined | F1 | Dice |
|---|---|---|---|
| 1-3 (tied) | 1.60 | 0.78 | 0.82 |
| 4 | 1.59 | 0.77 | 0.82 |
| 5-6 | 1.58 | 0.77 / 0.76 | 0.81 / 0.82 |
| 7 | 1.57 | 0.77 | 0.80 |
| 8 (last Phase 1 prize) | 1.56 | 0.75 | 0.81 |
| 11 | 1.53 | 0.73 | 0.80 |
| 35 | 1.02 | 0.39 | 0.63 |
| 50 | 0.79 | 0.17 | 0.62 |
Scores rounded to 2 decimals; ranks 1-8 span only 0.04. The top three all submitted on 19 Sep between 18:14 and 20:14. Dice has a high floor (~0.6); F1 discriminates. The organizers' baseline score is unknown (submit it as-is to calibrate). Check whether the per-submission detail page shows the natural/synthetic breakdown.

### 16.7 Compute and workflow decision (user input)
- User has Kaggle and Colab available; a local 2 GB GPU will NOT be used.
- Recommended: **Kaggle GPU for the baseline and main training** (the organizer code is written for Kaggle: `/kaggle/working` paths, Kaggle Secrets `HF_TOKEN`, Internet ON). Roughly 30 GPU-hours/week, T4x2 or P100, and about 29 GB RAM, versus about 12.7 GB RAM on free Colab [approximate figures from memory; check the account's quota page]. Colab as overflow.
- Workflow: edit in Antigravity, push to a private GitHub repo, pull into Kaggle notebooks. Cache decoded audio and labels to disk (e.g. a Kaggle dataset of int16 arrays) so decoding is not repeated every session.
- Phase 1 has under ~60 h left at the time of writing, so treat it as a calibration checkpoint. Phase 2 (17 Oct) carries far more prize money and the leaderboard is not reset.

### 16.8 User's listening notes (first-hand)
People from across India describe an image in their own language, indoors and outdoors, sometimes noisy and sometimes quiet. Noise includes road sounds, birds, coughs, etc. NOTE: even clips that sound quiet carry annotated events, mostly the speaker's own breaths, lip smacks and coughs (35% of events).

### 16.9 Planned experiments (no training needed unless noted)
1. **Union-oracle ceiling:** convert Gold labels to union events, score against raw Gold events. Tells us how much the union-only output costs.
2. **Boundary-jitter ceiling:** perturb Gold boundaries by 10 / 20 / 40 / 80 ms and measure F1 drop (effect of 20 ms frames, especially on short events).
3. **Trivial predictors:** "no events" and "whole clip" scored on Gold and on official validation. Checks the Dice floor and the empty-clip assumption.
4. **Per-tier / per-district / per-language statistics:** events per clip, share of whole-clip events, nested-event rate, class mix. Confirms the annotator-convention hypothesis for Silver.
5. **Faithful validation harness** (16.5), then **run the baseline as-is** and submit it once to calibrate.
6. **Inspect the synthetic validation clips** to reverse-engineer the mixing.
7. Design questions still open: how to output overlapping events (per-class runs, union plus ambient/transient split, or event-level prediction); per-district trust weights for Silver instead of a flat 0.4; whether forced alignment of Bronze/Silver transcripts (tags wrap the affected words) can supply or check timestamps.

---

## 16. Update log v2 (snapshot Sun 20 Sep 2026, ~00:15 IST). Supersedes earlier sections where they conflict.

### 16.1 Clock
The website countdown showed Phase 1 in 2d 11h 46m and Phase 2 in 27d 11h 46m. So about **60 h** remain until Phase 1 (22 Sep, 12:00 IST). Phase 2 ends 17 Oct, 12:00 IST.

### 16.2 Open questions from section 12: resolved
- **Bronze data:** it is in the same Hugging Face dataset. The single `train` split has 90,637 segments (~154.6 h), told apart by `annotationQuality`: `verified_timestamps` (Gold) 11,111 segs / ~21.8 h; `unverified_timestamps` (Silver) 61,642 / ~100.3 h; `no_timestamps` (Bronze) 17,884 / ~32.4 h. Bronze has `NoiseCategory` and tagged transcripts but no timestamps.
- **Validation location:** Codabench, Get Started → Files (released 4 Sep). The "planned HF release" wording is stale.
- **Held-out data:** 10 h of verified, speaker-disjoint audio is not in the HF release [STATED]. It is probably the natural part: 3 h validation + 7 h test [INFERRED]. Synthetic: 1 h validation + 4 h test. The website's "10 h gold-standard test" is consistent with this.
- **ΔWER convention [STATED, website announcement]:** Combined = SI-SDR + 100 × ΔWER **as a fraction**. The leaderboard shows ΔWER as a percent (e.g. −1.17) for display only, so 1 percentage point of WER = 1 point of Combined. The website's phrase "equal-weight average" is loose wording, and the Codabench formula is authoritative.
- **Model/data policy [STATED, FAQ]:** "open model and open data policy". Teams may use publicly available, closed-source or proprietary models, and additional data or augmentation. Terms still require external data and pretrained models to be **disclosed and documented**, and top-5 must release training and inference code with documented dependencies. A closed-source dependency at inference may conflict with the reproducibility requirement. Check with the organizers before relying on one.
- **Team rules [STATED, FAQ]:** anyone based in India, at least one member from an Indian institution or organization, 3–4 members typical, one team per person.
- **Whole-clip tags:** transcripts contain `<static noise>`, `<noise>`, `<talking>`, `<people talking>`, `<wind noise>`, `<thudding>`, but these never appear in the timestamp lists, so the label space is exactly the 7 classes.
- **Empty clips:** in the release-quality subset, 72,746 of 72,756 segments have at least one timestamped event, and every segment has at least one category. Test is probably similar [INFERRED]. Confirm on the validation file.
- **Still open:**
  - Does "final submission at the deadline" mean the last or the best submission?
  - Does Track 2 test input come with organizer-provided timestamps?
  - What is the leaderboard tie-break, given scores are rounded to 2 decimals?
  - Does the test set have the same language mix as training?
  - Is Silver's whole-clip labelling a convention that differs from Gold's?

### 16.3 Dataset facts (HF card, dataset paper arXiv 2609.02474, corpus paper 2603.28714)
- **Fields:** `audio`, `imageFileName`, `state`, `district`, `duration`, `language`, `annotationQuality`, `isTranscriptionAvailable`, `transcript`, `NoiseCategory` (list), `NoiseSubCategoryTimeStamp` (list of {category, tag, start, end}, with times as strings). **There is no speaker ID and no unique clip ID** (`imageFileName` repeats), so a speaker-disjoint split is not possible. Use `district` as a proxy. The dataset is gated (HF token needed), 17.7 GB of parquet, CC-BY-4.0.
- **Dataset paper (2 Sep 2026):** mean segment 6.05 s (0.79–23.49 s). Per-category events / event-hours: non-speech human 37,739 / 4.5 h (mean 0.42 s), animal 24,601 / 22.4 h, vehicle 20,603 / 12.5 h, baby-child 12,376 / 10.5 h, singing-music 6,978 / 9.7 h, phone-alarm 3,683 / 1.8 h, appliance 912 / 1.5 h. Event-hours sum to about 62.9 h out of 122.17 h, so noise covers up to ~51% of audio, less where events overlap. Single-category segments 55,330; multi-category 17,426. Hindi is ~84 h (~69%), then Telugu 16.7 h, Bengali 12.9 h, Marathi 5.3 h. It has no baselines or model results.
- **F1 and Dice see different data:** non-speech human events are 35% of events but only ~7% of noise time (F1-dominant). Animal, vehicle and music dominate noise time (Dice-dominant).
- **QC:** Silver = freelancer output that passed a sanity check. Gold = ≥20 h re-timestamped by an internal team using Silver as a reference, then a 10% independent audit, with the batch redone if even one sampled event disagrees. Test follows Gold conventions [INFERRED].
- **Corpus paper conventions:** segments ≥0.5 s. VAD checks flag silence over 0.3 s at the start or end, and over 1 s inside a segment. In transcripts `{...}` holds a corrected spelling (not noise), `--` marks an incomplete utterance, and `<PAUSE>` marks a pause over 0.5 s. Speaker-produced non-speech sounds (breathing, lip smacks, coughs) are transcribed as foreground events, so `human_non_speech` events are made by the speaker.
- **Observed in rows (a viewer sample of ~100, clustered by state, not random):** events covering the whole clip (0.000 to end) are common, e.g. `<insect noise>`, `<engine sound>`, `<vehicle noise>`, `<bird squawking>`. They are especially frequent in Silver rows from Washim, Purulia and Pathankot, but also appear in Gold (`<vehicle horn>` 0.000–5.819). Nested events are common (short `[breathing]` inside a whole-clip event). Transcript tags seem to wrap the words that overlap the noise, and an empty pair like `<horn> </horn>` marks an instant [hypotheses to check].
- **User listening notes:** speakers describe an image spontaneously in their native language (not reading), indoors and outdoors, with noise ranging from silent to busy (road, birds, coughs, etc.). Most clips have at least a short breath or lip smack.

### 16.4 Organizer baseline (repo `nagshubhadip/noise-event-detection-and-removal`)
- **Layout:** `track1_detection/versions/` has `v-1` ATST-Frame SED, `v-2` CRNN mean-teacher SSL, `v-3_indoml2026_track1` WavLM SED (mean-teacher + MIL). `track2_removal/versions/v-1_indoml2026_track2` has an STFT mask net with SraVaani/WavLM FiLM. Each version has the original notebook and a `modular/` package (`config, data, model, losses, train, inference, evaluate, main`). It targets Kaggle GPU (`/kaggle/working`, HF token via Kaggle Secrets). Files read so far are only for v-3 Track 1.
- **v-3 design:**
  - Model: `microsoft/wavlm-base-plus` (last 2 layers dropped) → 2-layer BiGRU (256) → a frame head (8 channels: "any noise" + 7 classes) and an attention-pooled clip head (MIL). Output is on a 20 ms grid.
  - Data and inputs: 5 s windows; per-clip standardization; random crop (validation: first 5 s); time-mask augmentation and mixup (p 0.5, union labels).
  - Losses: frame BCE with `pos_weight=4` (Gold and Silver only, Silver ×0.4, Bronze zeroed); clip-tag BCE on all tiers (unweighted); mean-teacher consistency (EMA 0.999, weight ≤2, 5-epoch ramp).
  - Training: two stages (encoder frozen, then unfrozen with encoder LR 5e-5 and head LR 1e-3). Validation is a random 10% of Gold.
  - Post-processing (`prob_to_events`): threshold → median filter → merge gaps ≤0.1 s → drop events <`min_dur`, plus a clip gate. It sweeps thr / med / min_dur / gate on validation.
  - Budgets: `fast` is a 100-clip smoke run; `full` (~7 h) caps Silver at 20k and Bronze at 8k clips. All audio is kept in RAM as int16.
- The scorer functions in `evaluate.py` are verbatim copies of the Codabench code.

### 16.5 Issues found in the baseline (from code reading; nothing run yet)
1. **Only the union channel is exported.** The 7 category channels are trained but unused, so nested or overlapping events collapse into one span.
2. **Local validation is not faithful to the leaderboard.** `cache_posteriors` rebuilds the reference from the rasterized 20 ms union mask, which merges nested or touching events. It also validates only the first 5 s of each clip, and the split is random over Gold, with no district or speaker separation.
3. **Streaming order:** `load_stream` does not shuffle, so the tier caps take the first rows, which may be geographically clumped.
4. **The `QW["no_timestamps"]` = 0.2 weight is never used.** `l_weak` is unweighted for all tiers.
5. **Mixup can mix a timestamped clip with a Bronze clip** without adjusting `strong` or `w`.
6. **`pos_weight=4` looks unjustified**, since noise covers ~50% of audio time, and it likely biases the model toward over-prediction.
7. **The union channel gets 1/8 of the frame loss**, and the loss has no explicit boundary term.
8. **Only the last encoder layer is used**, with no layer-weighted sum. Earlier layers may transfer better for non-speech sounds.
9. **One global threshold, median filter and minimum duration for all classes**, even though F1 is dominated by 0.4 s human sounds and Dice by 3–6 s animal, vehicle and music events.
10. **The "full" run uses only part of Silver and Bronze**, probably because of RAM.
11. **Checkpoints are selected on the flawed validation score** with fixed default post-processing.
12. **The clip gate is probably useless** if test clips are almost never empty.

### 16.6 Track 1 leaderboard snapshot (19 Sep ~22:46 IST; 70 teams, 50 shown)
- Ranks 1–3 tied at Combined 1.60 (F1 0.78, Dice 0.82). Rank 4: 1.59. Rank 8, the last Phase 1 prize position: 1.56 (F1 0.75, Dice 0.81). Rank 11: 1.53. Rank 35: 1.02. Rank 50: 0.79 (F1 0.17, Dice 0.62).
- Dice has a high floor (~0.6) and F1 does most of the discriminating.
- The top three submitted between 18:14 and 20:14 on 19 Sep. The organizers' baseline score is unknown, and our own score is unknown.

### 16.7 Decisions, preferences, and next steps
- **Compute:** prefer **Kaggle GPU** (the baseline targets it, and it has more RAM than free Colab). The user's local GPU (2 GB) cannot train and is CPU-analysis only. Colab is possible. Keep any cached data in a private Kaggle dataset (the terms forbid redistribution).
- **Working style:** slow and deliberate, both tracks, already registered.
- **Proposed order:**
  1. A metadata-only statistics script: per-tier event counts and durations, union noise fraction, whole-clip and nested event rates, empty clips, Gold district mix.
  2. A faithful validation harness: whole clips, raw reference events (not rasterized), district-based split.
  3. Label-only experiments: union-oracle F1 against raw events, boundary-jitter ceiling at 10/20/40/80 ms, and "no events" / "whole clip" trivial predictors.
  4. Submit the organizer baseline as-is for calibration.
  5. Only then choose the design for overlapping events, Silver weighting, and class-specific post-processing.
- **Phase 2 idea (unvalidated):** align transcript words to audio to recover approximate timestamps for Bronze and to check Silver labels. Needs multilingual forced alignment, so it is not a Phase 1 task.
- **Still to request from the user:** the remaining v-3 files if they matter, the user's Kaggle/GPU setup, the validation file structure and a few rows, and the baseline's leaderboard score once submitted.
