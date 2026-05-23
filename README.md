<div align="center">

# Kaggle &nbsp;·&nbsp; BirdCLEF+ 2026

### Inference-only submission · CPU · Pantanal soundscapes

[![Competition](https://img.shields.io/badge/Kaggle-BirdCLEF%2B%202026-111?style=for-the-badge&labelColor=000)](https://www.kaggle.com/competitions/birdclef-2026)
[![License](https://img.shields.io/badge/Code-Apache%202.0-111?style=for-the-badge&labelColor=000)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-111?style=for-the-badge&labelColor=000)](#)
[![Runtime](https://img.shields.io/badge/Runtime-CPU%20only-111?style=for-the-badge&labelColor=000)](#runtime)
[![Public LB](https://img.shields.io/badge/Public%20LB-0.914-111?style=for-the-badge&labelColor=000)](#experiments-public-leaderboard)

**`Perch 2.0 (ONNX)`** &nbsp;·&nbsp; **`4× ProtoSSM`** &nbsp;·&nbsp; **`Student CNN`** &nbsp;·&nbsp; **`Student CRNN`** &nbsp;·&nbsp; **`Site/Hour prior`**

</div>

---

## Overview

CPU-only Kaggle submission for the [BirdCLEF+ 2026 competition](https://www.kaggle.com/competitions/birdclef-2026): 234 species, Pantanal (Brazilian wetland) soundscapes, 90-minute CPU runtime budget at scoring time.

This repository contains **only my own code and configuration**. It is **inference-only**: no model is trained or fine-tuned here. The pre-trained artifacts (Google Perch 2.0, the chaneyma MoE bundle, and the rishikeshjani ONNX export of Perch) are used as released and are **not** redistributed in this repository — see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

## Method

The submission ensembles a frozen audio foundation model with a small mixture of experts:

- **Perch 2.0 (frozen teacher)** — Google's bioacoustics foundation model, used via an ONNX export for CPU inference. Outputs per-class logits plus a 1536-dim embedding.
- **chaneyma MoE** — 4× ProtoSSM folds (selective state-space + class prototypes, consuming the Perch embedding) plus a Student CNN and a Student CRNN on log-mel features.
- **Site / hour prior** — empirical priors over the 234 classes conditioned on recording site (`S\d+` parsed from filename) and hour-of-day (24 bins), fit from the BirdCLEF+ 2026 `train_soundscapes_labels.csv` only.
- **Temporal smoothing** — per file, over adjacent 5-second windows (12 windows per 60-second file).

### My contributions

1. **ONNX patch of Perch.** The upstream chaneyma inference script (CC0) loaded Perch through `tf.saved_model.load`, which requires TensorFlow and was too heavy for the Kaggle CPU budget. I replaced the TF code path with `onnxruntime` plus a CC0 ONNX export of Perch (rishikeshjani), removing the TensorFlow dependency entirely.
2. **Blend-weight search** over `(perch, student_cnn, student_crnn)` mixing weights of the pre-sigmoid logits.
3. **Prior-scale sweep** over the multiplier applied to the site/hour log-odds prior before it is added to the Perch logits.
4. **Per-file temporal smoothing** averaging each 5-second window with its immediate neighbours.

> Inference script originally by chaneyma (CC0), patched for ONNX/CPU inference by Serghei Brinza.

## Experiments (public leaderboard)

Metric: macro-averaged ROC-AUC (the competition metric — ranking-based).

| Variant | Blend (P / CNN / CRNN) | `--prior-scale` | Postprocessing | Public LB |
|---|---|---|---|---|
| Prior sweep | 0.80 / 0.13 / 0.07 | 0.60 | smoothing 0.8 / 0.1 / 0.1 | — |
| Prior sweep | 0.80 / 0.13 / 0.07 | 0.40 | smoothing 0.8 / 0.1 / 0.1 | — |
| Prior sweep | 0.80 / 0.13 / 0.07 | **0.20** | smoothing 0.8 / 0.1 / 0.1 | **0.914** |
| Prior sweep | 0.80 / 0.13 / 0.07 | **0.10** | smoothing 0.8 / 0.1 / 0.1 | **0.914** |
| Power transform on probs | 0.80 / 0.13 / 0.07 | 0.20 | + power transform | no change |
| Alt smoothing | 0.80 / 0.13 / 0.07 | 0.20 | smoothing 0.7 / 0.15 / 0.15 | 0.913 |

**Best final public-LB score: 0.914.**

Honest caveats:

- The power transform on probabilities had no effect because the macro ROC-AUC metric is ranking-based.
- The 0.7 / 0.15 / 0.15 temporal-smoothing variant was *slightly worse* than the shipped 0.8 / 0.1 / 0.1 version. Reporting it because reporting unsuccessful variations is the point of an honest experiment log.
- Intermediate prior-scale rows (0.60, 0.40) are left blank rather than filled in with guessed numbers.

## Runtime

- **CPU only.** Perch runs on CPU via `onnxruntime`; ProtoSSM, Student CNN and Student CRNN run on CPU via PyTorch. There is no GPU code path.
- **Kaggle 90-minute CPU budget.** Designed around the BirdCLEF+ 2026 scoring environment.
- **Audio**: 32 kHz, 60-second `.ogg` test files, processed as 12 non-overlapping 5-second windows per file.
- **Classes**: 234, taken from the competition `sample_submission.csv`.
- **Region**: Brazilian Pantanal soundscapes.

## How to run

```bash
python src/infer_moe_onnx.py \
  --blend-perch 0.80 \
  --blend-cnn   0.13 \
  --blend-crnn  0.07 \
  --prior-scale 0.20 \
  --out submission.csv
```

The script exposes **15 CLI flags** in total (paths to artifact directories, fold weight prefix, legacy single-student fallback, embedding dim, etc.). Run `python src/infer_moe_onnx.py --help` for the full list. Only the five flags shown above were varied during experiments; the rest stayed at their defaults.

You will also need to download the pre-trained artifacts yourself from the sources listed in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) and arrange them under the paths the defaults expect (or override via the path flags).

## Considered but not pursued

A short, honest design-space review. These directions are listed because I read about them while preparing this submission; **I did not run any of them in this work.**

- **Audio foundation models beyond Perch** — BirdMAE, NatureLM-audio. Potentially stronger embeddings but unclear whether they fit the 90-minute CPU budget without distillation work I did not want to claim.
- **Semi-supervised distillation** from a larger teacher into a smaller CPU student. Out of scope here (no training in this repo).
- **SED (sound-event-detection) heads** with frame-wise localisation rather than per-window logits. Would change the I/O contract; not pursued.

## References

- van Merrienboer, B. et al. (2025). *Perch 2.0.* arXiv:2508.04665.
- Sydorskyi, V. & Goncalves, F. (2025). *BirdCLEF+ 2025 — 2nd-place CLEF Working Note.* CEUR Workshop Proceedings, Vol. 4038. (Related-work reference; not used as code or data here.)

A machine-readable version is in [CITATION.cff](CITATION.cff).

## Licensing

- My code in this repository: **Apache License 2.0** — see [LICENSE](LICENSE).
- Pre-trained artifacts and the upstream inference script: see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md). They are **not** redistributed here.

---

<div align="center">

<sub>Independent submission. Not affiliated with or endorsed by Kaggle, Google, or the BirdCLEF organizers.</sub>

</div>
