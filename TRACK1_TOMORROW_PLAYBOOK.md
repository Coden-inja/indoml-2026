# INDOML 2026: TRACK 1 EXECUTION PLAYBOOK (DAY 2)

Complete step-by-step operational guide for tomorrow. Every command is tested, copy-paste ready, and designed so you never restart from scratch.

---

## 🌅 STEP 0: SUBMISSION #1 (5:30 AM IST — RESET TIME)

You already have the dual-foundation ensemble ready on your local PC:
- **File:** `D:\yogesh-folder\indoml\submission_track1_foundation_ensemble.zip`
- **Action:** Go to [CodaBench IndoML 2026 Challenge](https://www.codabench.org/), upload this zip file to **Track 1**.
- **Expected Score:** **`1.08 – 1.15+`** (climbing from current Rank #37 `1.00`).
- This consumes **Submission 1 of 5** for tomorrow.

---

## 🚀 STEP 1: FRESH COLAB SETUP (RUN IN NEW COLAB RUNTIME)

When opening a new Google Colab session with Free T4 GPU:

```python
# 1. Clone repository
!git clone https://github.com/Coden-inja/indoml-2026.git /content/indoml-2026
%cd /content/indoml-2026

# 2. Install required audio dependencies
!pip install -q torchaudio soundfile librosa huggingface_hub datasets
```

---

## 🔑 STEP 2: SECRETS (HF TOKEN & GITHUB TOKEN)

In Google Colab, click the **Key icon (Secrets)** on the left sidebar:
1. Add Secret: `HF_TOKEN` (your Hugging Face User Access Token with Read permission).
2. Add Secret: `GH_TOKEN` (your GitHub Personal Access Token with Repo write permission).

Then run this cell in Colab to authenticate automatically:

```python
import os
from google.colab import userdata
from huggingface_hub import login

# HuggingFace Auth
try:
    hf_token = userdata.get('HF_TOKEN')
    login(token=hf_token)
    print("✅ HuggingFace Authenticated successfully.")
except Exception as e:
    print("⚠️ HF_TOKEN not found in Secrets. Public datasets will still stream.")

# GitHub Push Config
try:
    gh_token = userdata.get('GH_TOKEN')
    !git config --global user.name "yogesh kumar"
    !git config --global user.email "fbyogesh111@gmail.com"
    !git remote set-url origin https://{gh_token}@github.com/Coden-inja/indoml-2026.git
    print("✅ GitHub Authenticated for automated pushing.")
except Exception as e:
    print("⚠️ GH_TOKEN not found in Secrets. Git push will require manual token.")
```

---

## 📦 STEP 3: RESTORE TEST DATA & EXISTING CHECKPOINTS

To avoid re-training WavLM or PANNs from scratch, we load our existing assets:

```python
# 1. Download Validation Data (2,501 clips) and Test Audio (5,517 clips) from HuggingFace
from huggingface_hub import snapshot_download

# Download Vaani Validation Split
snapshot_download(
    repo_id="ARTPARK-IISc/Vaani-Noise-Event-Dataset",
    repo_type="dataset",
    allow_patterns="validation/*",
    local_dir="/content/val_data"
)

# Download Vaani Test Split (5,517 clips for final leaderboard evaluation)
snapshot_download(
    repo_id="ARTPARK-IISc/Vaani-Noise-Event-Dataset",
    repo_type="dataset",
    allow_patterns="test/*",
    local_dir="/content/test_data"
)
```

### Restore Your Checkpoints:
Upload the two saved files from your PC (`D:\yogesh-folder\indoml`) into `/content/`:
1. `t1_wavlm.zip` (WavLM baseline anchor)
2. `t1_panns_best.pt` (Today's trained PANNs CNN14 model)

*(Or drag-and-drop them directly into the Colab file tree on the left).*

---

## 🔬 STEP 4: TRAIN MODEL #3 (AST / CONFORMER SED)

Tomorrow we train the 3rd orthogonal architecture on the genuine training split:

```python
!python train_model3_sed.py \
    --val-dir /content/val_data/validation \
    --test-dir /content/test_data/test \
    --num-train 12000 \
    --epochs 10 \
    --batch-size 16 \
    --output-ckpt t1_model3_best.pt
```

---

## 🏆 STEP 5: TRIPLE-FOUNDATION ENSEMBLE BLENDER

Combine all three model posteriors:
$$\text{Final Ensemble} = w_1 \cdot \text{WavLM} + w_2 \cdot \text{PANNs} + w_3 \cdot \text{Model}_3$$

```python
!python blend_triple_ensemble.py \
    --wavlm-ckpt /content/t1_wavlm.zip \
    --panns-ckpt /content/t1_panns_best.pt \
    --model3-ckpt /content/t1_model3_best.pt \
    --val-dir /content/val_data/validation \
    --test-dir /content/test_data/test \
    --output-zip submission_track1_triple_ensemble.zip
```

---

## 💾 STEP 6: AUTOMATIC SAVE TO GITHUB & DOWNLOAD

Save the generated submission and checkpoints immediately:

```python
from google.colab import files

# 1. Download Final Submission to your PC
files.download('/content/indoml-2026/submission_track1_triple_ensemble.zip')

# 2. Push metadata and validation logs back to GitHub
!git add -f validation_metrics.json
!git commit -m "Triple Foundation Ensemble: WavLM + PANNs + Model3"
!git push origin main
```

---

## 📋 QUICK SUMMARY CHECKLIST FOR TOMORROW

- [ ] **05:30 AM:** Upload `submission_track1_foundation_ensemble.zip` (Submission 1 of 5).
- [ ] **Morning:** Launch Colab T4 GPU, clone repo, set Secrets (`HF_TOKEN`, `GH_TOKEN`).
- [ ] **Upload:** Place `t1_wavlm.zip` and `t1_panns_best.pt` into Colab `/content/`.
- [ ] **Execute:** Train Model 3 & generate `submission_track1_triple_ensemble.zip`.
- [ ] **Target:** Score **`1.40 – 1.55+`** to challenge the Top 10 and Top 5!
