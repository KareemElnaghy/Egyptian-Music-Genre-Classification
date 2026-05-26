# Egyptian Music Genre Classifier

A deep learning system that automatically classifies Egyptian music into five genres — **Tarab, Egyptian Pop, Mahraganat, Shaabi, and Egyptian Rap** — using a transfer-learned CNN14 backbone fused with hand-crafted acoustic features.

---

## Contributors
- [Farida Bey](https://github.com/FaridaBey)
- [Kareem Elnaghy](https://github.com/KareemElnaghy)


## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Goal](#goal)
3. [Our Contributions](#our-contributions)
4. [Architecture Overview](#architecture-overview)
5. [Dataset](#dataset)
6. [Results](#results)
7. [Project Structure](#project-structure)
8. [Building & Running](#building--running)
9. [Training the Model](#training-the-model)
10. [Explainability (Grad-CAM)](#explainability-grad-cam)
11. [Deliverables](#deliverables)

---

## Problem Statement

Most MIR research focuses on Western music (GTZAN: rock, jazz, blues). Egyptian/Eastern  genres remain completely absent from labeled datasets. State-of-the-art models are "culturally blind" trained on Western scales, harmonies, and rhythms. They fail on: Microtonal maqam scales, Tabla rhythmic patterns, Unique Egyptian vocal timbres.

---

## Goal

Build an accurate, interpretable, and deployable genre classifier for Egyptian music that:

- Covers five culturally significant Egyptian genres (Tarab, Mahraganat, Shaabi, Egyptian Pop, Egyptian Rap)
- Leverages a large-scale pretrained audio backbone (CNN14 / PANNs) via transfer learning
- Enriches deep features with hand-crafted acoustic descriptors specific to Egyptian music (chroma, MFCCs, spectral contrast, zero-crossing rate)
- Provides visual explanations via Grad-CAM so predictions are not a black box
- Is accessible to anyone through a web interface — no technical knowledge required

---

## Our Contributions

| Contribution | Description |
|---|---|
| **Custom Dataset** | Collected a novel, labeled Egyptian music dataset from YouTube using genre-specific Arabic and English queries via `yt-dlp`. Each track is segmented into 30-second WAV clips at 32 kHz. |
| **Dual-Branch Architecture** | Designed a two-branch model that fuses CNN14's 2048-dim learned embedding with a 64-dim hand-crafted feature MLP, yielding richer representations than either branch alone. |
| **Channel Attention** | Added a CBAM-style channel attention module after `conv_block6` of CNN14, allowing the network to selectively weight the most informative frequency channels. |
| **Domain-Aware Feature Engineering** | Extracted a 72-dimensional feature vector per clip: Chroma STFT (24), MFCCs (40), Spectral Contrast (7), and Zero Crossing Rate (1) — descriptors well-suited to the tonal and rhythmic properties of Egyptian music. |
| **Augmentation Pipeline** | Combined SpecAugment (frequency/time masking), pitch shifting, time stretching, additive noise, and Mixup to improve generalization from a limited dataset. |
| **Grad-CAM Explainability** | Implemented Grad-CAM targeting `conv_block6` to produce time-frequency saliency maps, making the model's reasoning transparent and inspectable. |
| **Production Web App** | Built a FastAPI backend with a styled frontend that accepts `.wav` or `.mp3` uploads and returns a genre prediction in real time. |

---

## Architecture Overview

<img width="377" height="441" alt="image" src="https://github.com/user-attachments/assets/5b862e96-8bee-4cd9-b9d2-3544ce56374e" />


**Key hyperparameters (best config):**

| Parameter | Value |
|---|---|
| Backbone | CNN14 (PANNs, pretrained on AudioSet) |
| Frozen layers | `partial3` — last 3 conv blocks unfrozen |
| Backbone LR multiplier | 0.05 |
| Learning rate | 4.63e-3 |
| Optimizer | AdamW + CosineAnnealingLR |
| Batch size | 32 |
| Max epochs | 80 (early stopping patience: 15) |
| Early stopping metric | Macro F1 |
| Fusion hidden dim | 256 |
| Mixup alpha | 0.126 |
| Sample rate | 32,000 Hz |
| Clip duration | 30 seconds |

---

## Dataset

- **5 genres**: Tarab, Egyptian Pop, Mahraganat, Shaabi, Egyptian Rap
- **Collection method**: `collect_music.py` — searches YouTube with genre-specific Arabic/English queries, downloads with `yt-dlp`, segments into 30-second clips, exports as 32 kHz mono WAV
- **Split**: Song-level stratified split — **70% train / 15% val / 15% test** — ensuring no song appears in more than one split
- **Metadata**: `dataset/metadata.csv` — maps each clip filename to its genre and source song
- **Test set size**: 148 clips across all five genres

---

## Results

### Test Set Performance (93% Accuracy)

| Genre | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Tarab | 0.95 | 1.00 | **0.97** | 19 |
| Egyptian Pop | 0.88 | 1.00 | **0.93** | 21 |
| Mahraganat | 0.92 | 0.95 | **0.94** | 38 |
| Shaabi | 0.92 | 0.83 | **0.88** | 42 |
| Egyptian Rap | 1.00 | 0.96 | **0.98** | 28 |
| **Macro avg** | **0.93** | **0.95** | **0.94** | 148 |
| **Weighted avg** | **0.93** | **0.93** | **0.93** | 148 |

**Overall accuracy: 93%** on the held-out test set.

---

## Project Structure

```
Final Milestone/
├── app/                        # Web application
│   ├── main.py                 # FastAPI server
│   ├── requirements.txt        # Runtime dependencies
│   ├── static/                 # Frontend (HTML, CSS, JS)
│   └── Final/
│       ├── src/predict.py      # Inference pipeline
│       └── trained_model/      # Saved model weights + scaler
│           ├── final_trained_model.pth
│           └── final_scaler.pkl
├── src/                        # Training source code
│   ├── model.py                # Transfer_Cnn14 architecture
│   ├── dataset.py              # EgyptianMusicDataset + splits
│   ├── features.py             # Hand-crafted feature extraction
│   ├── augmentation.py         # SpecAugment + audio augmentations
│   ├── train.py                # Training loop (single-split & k-fold)
│   ├── evaluate.py             # Evaluation & metrics
│   ├── explain.py              # Grad-CAM implementation
│   └── plot_curves.py          # Training curve plots
├── checkpoints/
│   └── Cnn14_mAP=0.431.pth    # CNN14 AudioSet pretrained checkpoint
├── dataset/                    # Audio clips + metadata
│   ├── metadata.csv
│   ├── Egyptian Pop/
│   ├── Egyptian Rap/
│   ├── Mahraganat/
│   ├── Shaabi/
│   └── Tarab/
├── website/                    # Project report website
│   └── team-specific/index.html
├── collect_music.py            # YouTube data collection script
├── training_demo.py            # End-to-end train + evaluate demo
└── demo/                       # Demo run outputs
    └── data/split.json         # Fixed train/val/test song split
```

---

## Building & Running

### Prerequisites

- Python 3.9+
- pip + a virtual environment (recommended)

### 1. Set up the environment

```bash
python -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

pip install -r app/requirements.txt
pip install pyyaml pandas matplotlib tqdm   # extra deps for training
```

### 2. Download the CNN14 pretrained checkpoint

The CNN14 AudioSet checkpoint is required for both training and the web app (if running from scratch). The checkpoint is already included at `checkpoints/Cnn14_mAP=0.431.pth`. If it is missing, download it from [Zenodo (record 3987831)](https://zenodo.org/record/3987831) and place it at that path.

### 3. Run the web app

The trained model weights are already bundled in `app/Final/trained_model/`, so you can run the app immediately without training:

```bash
uvicorn app.main:app --reload
```

Then open [http://localhost:8000](http://localhost:8000) in your browser. Upload any `.wav` or `.mp3` file (up to 50 MB) and the app will return the predicted Egyptian music genre.

---

## Training the Model

### Quick demo (recommended first step)

Retrains the model from scratch using the best config and evaluates on the held-out test set. Results land in `demo/`:

```bash
python training_demo.py
```

After training finishes, plot the loss/accuracy curves:

```bash
python src/plot_curves.py
```

### Collecting new data

To download audio clips for a specific genre:

```bash
python collect_music.py --genre Mahraganat --num_clips 100 --output_dir dataset_new
```

The script uses `yt-dlp` (must be installed separately: `pip install yt-dlp`) and searches YouTube with genre-specific Arabic and English queries.

---

## Explainability (Grad-CAM)

Grad-CAM heatmaps are generated by `src/explain.py`, targeting `conv_block6` of the CNN14 backbone. Each heatmap shows which time-frequency regions of the log-mel spectrogram most influenced the model's prediction.

---

## Deliverables

| Deliverable | Location / Description |
|---|---|
| **Web App** | `app/` — FastAPI server with a styled browser UI for real-time genre prediction. Run with `uvicorn app.main:app --reload`. |
| **Project Website** | `website/team-specific/index.html` — Full project report with architecture diagrams, spectrogram examples, confusion matrix, Grad-CAM results, and team information. |
| **Trained Model** | `app/Final/trained_model/final_trained_model.pth` + `final_scaler.pkl` — ready-to-use weights and feature scaler. |
| **Training Results** | Classification report, training log (CSV), confusion matrix (normalized), training curves, and per-genre Grad-CAM visualizations. |
| **Poster** | Presented at the course final showcase summarizing the problem, methodology, architecture, and results. |
| **Presentation** | Slide deck covering motivation, related work, architecture design decisions, ablation experiments, results, and demo. |
| **Live Demo** | Demonstrated live at the final showcase using the web app deployed locally, classifying real Egyptian music clips on the spot. |
| **Source Code** | This repository — fully reproducible from data collection through training, evaluation, and deployment. |

---

## Project Proposal and Progress Presentations
You can find them [here](https://drive.google.com/drive/folders/1OvfEHxLaQnC_NOqzey5Tmifq1I5wHs97?usp=sharing)
## Dependencies

**Runtime (app/requirements.txt):**

```
fastapi
uvicorn
python-multipart
torch >= 1.12.0
torchlibrosa >= 0.1.0
librosa >= 0.9.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
soundfile >= 0.10.0
timm >= 0.6.0
scipy >= 1.7.0
```

**Additional (training only):**

```
pyyaml
pandas
matplotlib
tqdm
yt-dlp          # for collect_music.py only
```
