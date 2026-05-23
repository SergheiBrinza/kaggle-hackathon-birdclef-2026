# Third-Party Licenses and Provenance

The code in this repository (everything not listed below) is original work by Serghei Brinza, released under the Apache License 2.0 — see [LICENSE](LICENSE).

The pre-trained artifacts and upstream code listed here are **not redistributed** in this repository. You must download them yourself from the original sources, where they remain under their respective licenses.

## Inference script provenance

`src/infer_moe_onnx.py` was originally published by Kaggle user **chaneyma** under CC0 1.0 as part of the MoE artifacts dataset linked below. It has been patched for ONNX-based, TensorFlow-free CPU inference of the Perch teacher by **Serghei Brinza**. The patched file remains compatible with CC0; modifications added in this repository are additionally covered by the repository's Apache-2.0 license.

> Inference script originally by chaneyma (CC0), patched for ONNX/CPU inference by Serghei Brinza.

## Pre-trained artifacts (not redistributed)

| Component | Source | License |
|---|---|---|
| Google Perch 2.0 (frozen teacher) | https://huggingface.co/cgeorgiaw/Perch | Apache License 2.0 |
| Perch ONNX export for BirdCLEF+ 2026 | https://www.kaggle.com/datasets/rishikeshjani/perch-onnx-for-birdclef-2026 | CC0 1.0 |
| chaneyma MoE artifacts (4× ProtoSSM folds + StudentCNN + StudentCRNN) | https://www.kaggle.com/datasets/chaneyma/birdclef-2026-cv9245-moe-artifacts | CC0 1.0 |

These artifacts are used **as released**: no fine-tuning, distillation, or re-training was performed in this submission.

## Notes

- Perch 2.0 is © Google LLC, released by the Perch authors under Apache 2.0.
- The Perch ONNX export and the chaneyma MoE artifacts are released under CC0 1.0 by their respective Kaggle authors, which permits the use described here without additional permission.
- This repository does not reproduce, mirror, or imitate the Kaggle or BirdCLEF logos. The project icon in `assets/icon.svg` is an original work.
