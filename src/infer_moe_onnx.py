#!/usr/bin/env python3
from __future__ import annotations

# Derived from chaneyma's pantanal_infer_only_submission.py (CC0 1.0).
# Modified by Serghei Brinza: Perch teacher ported from TensorFlow SavedModel to ONNX Runtime (CPU). 2026.
# This modified file is licensed under Apache-2.0 (see LICENSE); upstream artifacts retain their own licenses (see THIRD_PARTY_LICENSES.md).

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
# tensorflow removed - using ONNX
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES = 60 * SR
N_WINDOWS = 12
FNAME_RE = re.compile(r"BC\d+_(?:Train|Test)_\d+_(S\d+)_")


def parse_site(filename: str) -> str:
    m = FNAME_RE.match(filename)
    return m.group(1) if m else "S00"


def parse_hour(filename: str) -> int:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 6:
        return 0
    hhmmss = parts[-1]
    if len(hhmmss) >= 2 and hhmmss[:2].isdigit():
        return int(hhmmss[:2]) % 24
    return 0


def read_60s_windows(path: Path) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != SR:
        raise ValueError(f"bad sr {sr} for {path}")
    if len(wav) < FILE_SAMPLES:
        wav = np.pad(wav, (0, FILE_SAMPLES - len(wav)))
    elif len(wav) > FILE_SAMPLES:
        wav = wav[:FILE_SAMPLES]
    return wav.reshape(N_WINDOWS, WINDOW_SAMPLES)


def fit_simple_prior_tables(y_flat: np.ndarray, sites_flat: np.ndarray, hours_flat: np.ndarray) -> Dict:
    eps = 1e-4
    global_p = np.clip(y_flat.mean(axis=0).astype(np.float32), eps, 1.0 - eps)

    site_to_stats = {}
    for s in sorted(set(sites_flat.tolist())):
        m = sites_flat == s
        if m.sum() == 0:
            continue
        site_to_stats[str(s)] = (int(m.sum()), np.clip(y_flat[m].mean(axis=0), eps, 1.0 - eps).astype(np.float32))

    hour_to_stats = {}
    for h in range(24):
        m = hours_flat == h
        if m.sum() == 0:
            continue
        hour_to_stats[int(h)] = (int(m.sum()), np.clip(y_flat[m].mean(axis=0), eps, 1.0 - eps).astype(np.float32))

    return {"global_p": global_p, "site_to_stats": site_to_stats, "hour_to_stats": hour_to_stats}


def prior_logits_from_tables(sites_flat: np.ndarray, hours_flat: np.ndarray, tables: Dict) -> np.ndarray:
    p = np.repeat(tables["global_p"][None, :], len(sites_flat), axis=0).astype(np.float32, copy=True)
    for i, (s, h) in enumerate(zip(sites_flat.tolist(), hours_flat.tolist())):
        if str(s) in tables["site_to_stats"]:
            n_s, p_s = tables["site_to_stats"][str(s)]
            w_s = n_s / (n_s + 8.0)
            p[i] = (1.0 - w_s) * p[i] + w_s * p_s
        if int(h) in tables["hour_to_stats"]:
            n_h, p_h = tables["hour_to_stats"][int(h)]
            w_h = n_h / (n_h + 8.0)
            p[i] = (1.0 - w_h) * p[i] + w_h * p_h
    p = np.clip(p, 1e-4, 1.0 - 1e-4)
    return (np.log(p) - np.log1p(-p)).astype(np.float32)


def postprocess_probs_filewise(probs_flat: np.ndarray, n_windows: int = N_WINDOWS) -> np.ndarray:
    n_rows, n_classes = probs_flat.shape
    assert n_rows % n_windows == 0
    x = probs_flat.reshape(-1, n_windows, n_classes).copy()
    prev_x = np.concatenate([x[:, :1, :], x[:, :-1, :]], axis=1)
    next_x = np.concatenate([x[:, 1:, :], x[:, -1:, :]], axis=1)
    x = 0.8 * x + 0.1 * (prev_x + next_x)
    top2 = np.sort(x, axis=1)[:, -2:, :].mean(axis=1, keepdims=True)
    x = x * top2
    return np.clip(x.reshape(n_rows, n_classes), 0.0, 1.0).astype(np.float32)


class SelectiveSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(d_model, d_model, d_conv, padding=d_conv - 1, groups=d_model)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, tlen, d_model = x.shape
        xz = self.in_proj(x)
        x_ssm, _ = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :tlen].transpose(1, 2)
        x_conv = F.silu(x_conv)
        dt = F.softplus(self.dt_proj(x_conv))
        A = -torch.exp(self.A_log)
        B = self.B_proj(x_conv)
        C = self.C_proj(x_conv)

        h = torch.zeros(bsz, d_model, self.d_state, device=x.device)
        ys = []
        for t in range(tlen):
            dt_t = dt[:, t, :]
            dA = torch.exp(A[None, :, :] * dt_t[:, :, None])
            dB = dt_t[:, :, None] * B[:, t, None, :]
            h = h * dA + x[:, t, :, None] * dB
            y_t = (h * C[:, t, None, :]).sum(-1)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)
        return y + x * self.D[None, None, :]


class ProtoSSM(nn.Module):
    def __init__(
        self,
        d_input: int = 1536,
        d_model: int = 320,
        d_state: int = 24,
        n_layers: int = 3,
        n_classes: int = 234,
        n_windows: int = 12,
        n_sites: int = 32,
        meta_dim: int = 16,
        dropout: float = 0.12,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_windows = n_windows
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)
        self.ssm_fwd = nn.ModuleList([SelectiveSSM(d_model, d_state) for _ in range(n_layers)])
        self.ssm_bwd = nn.ModuleList([SelectiveSSM(d_model, d_state) for _ in range(n_layers)])
        self.merge = nn.ModuleList([nn.Linear(2 * d_model, d_model) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.prototypes = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        self.proto_temp = nn.Parameter(torch.tensor(5.0))
        self.class_bias = nn.Parameter(torch.zeros(n_classes))
        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))

    def forward(self, emb: torch.Tensor, teacher_logits: torch.Tensor, site_ids: torch.Tensor, hours: torch.Tensor):
        bsz, tlen, _ = emb.shape
        h = self.input_proj(emb) + self.pos_enc[:, :tlen, :]
        s_emb = self.site_emb(site_ids)
        h_emb = self.hour_emb(hours)
        h = h + self.meta_proj(torch.cat([s_emb, h_emb], dim=-1))[:, None, :]

        for fwd, bwd, merge, norm in zip(self.ssm_fwd, self.ssm_bwd, self.merge, self.norms):
            res = h
            h_f = fwd(h)
            h_b = bwd(h.flip(1)).flip(1)
            h = norm(res + self.drop(merge(torch.cat([h_f, h_b], dim=-1))))

        h_norm = F.normalize(h, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        sim = torch.matmul(h_norm, p_norm.T) * F.softplus(self.proto_temp) + self.class_bias[None, None, :]
        alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]
        return alpha * sim + (1 - alpha) * teacher_logits


class StudentCNN(nn.Module):
    def __init__(self, emb_dim: int = 1536, n_classes: int = 234):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.emb_head = nn.Linear(512, emb_dim)
        self.logit_head = nn.Linear(512, n_classes)

    def forward(self, x: torch.Tensor):
        h = self.fc(self.body(x))
        emb = self.emb_head(h)
        logits = self.logit_head(h)
        return emb, logits


class StudentCRNN(nn.Module):
    def __init__(self, emb_dim: int = 1536, n_classes: int = 234):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
        )
        self.gru = nn.GRU(input_size=64 * 32, hidden_size=192, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(384, 384), nn.ReLU(inplace=True), nn.Dropout(0.2))
        self.emb_head = nn.Linear(384, emb_dim)
        self.logit_head = nn.Linear(384, n_classes)

    def forward(self, x: torch.Tensor):
        z = self.conv(x)
        b, c, m, t = z.shape
        z = z.permute(0, 3, 1, 2).reshape(b, t, c * m)
        out, _ = self.gru(z)
        h = self.fc(out.mean(dim=1))
        emb = self.emb_head(h)
        logits = self.logit_head(h)
        return emb, logits


class PerchTeacher:
    def __init__(self, teacher_dir: Path, taxonomy_csv: Path, sample_submission_csv: Path):
        import onnxruntime as ort
        # teacher_dir now points at the ONNX folder
        onnx_path = teacher_dir / "perch_v2.onnx"
        labels_path = teacher_dir / "labels.csv"
        
        self.sess = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.sess.get_inputs()[0].name

        taxonomy = pd.read_csv(taxonomy_csv)
        sample_sub = pd.read_csv(sample_submission_csv)
        self.primary_labels = sample_sub.columns[1:].tolist()

        labels_csv = pd.read_csv(labels_path)
        if "inat2024_fsd50k" in labels_csv.columns:
            labels_csv = labels_csv.rename(columns={"inat2024_fsd50k": "scientific_name"})
        elif "scientific_name" not in labels_csv.columns:
            labels_csv = labels_csv.rename(columns={labels_csv.columns[0]: "scientific_name"})

        labels_csv = labels_csv.reset_index().rename(columns={"index": "bc_index"})
        merged = taxonomy[["primary_label", "scientific_name"]].merge(
            labels_csv[["scientific_name", "bc_index"]], on="scientific_name", how="left"
        )
        bc_map = merged.set_index("primary_label")["bc_index"].to_dict()
        bc_indices = np.array(
            [int(bc_map.get(lbl)) if pd.notna(bc_map.get(lbl)) else -1 for lbl in self.primary_labels],
            dtype=np.int32,
        )
        mapped = bc_indices >= 0
        self.mapped_pos = np.where(mapped)[0].astype(np.int32)
        self.mapped_bc_idx = bc_indices[mapped].astype(np.int32)

    def infer(self, x: np.ndarray):
        # ONNX inference instead of TF SavedModel
        outputs = self.sess.run(None, {self.input_name: x.astype(np.float32)})
        # Perch ONNX outputs: [embedding, spatial_embedding, spectrogram, label]
        emb = outputs[0].astype(np.float32)  # (batch, 1536)
        raw_logits = outputs[3].astype(np.float32)  # (batch, 14795)
        scores = np.full((len(x), len(self.primary_labels)), -8.0, dtype=np.float32)
        scores[:, self.mapped_pos] = raw_logits[:, self.mapped_bc_idx]
        return scores, emb

def build_training_priors(data_2026_dir: Path, primary_labels: List[str]) -> Tuple[Dict, Dict[str, int]]:
    labels = pd.read_csv(data_2026_dir / "train_soundscapes_labels.csv")
    c2i = {c: i for i, c in enumerate(primary_labels)}

    rows = []
    for _, r in labels.iterrows():
        fn = str(r["filename"])
        end_sec = int(pd.to_timedelta(r["end"]).total_seconds())
        rows.append((fn, end_sec, str(r["primary_label"])))
    clean = pd.DataFrame(rows, columns=["filename", "end_sec", "primary_label"])
    clean = (
        clean.groupby(["filename", "end_sec"], as_index=False)["primary_label"]
        .apply(lambda x: ";".join(sorted(set(";".join(x).split(";")))))
    )

    files = sorted(clean["filename"].unique().tolist())
    y = np.zeros((len(files), N_WINDOWS, len(primary_labels)), dtype=np.float32)

    by_file = {fn: {int(r.end_sec): str(r.primary_label) for r in g.itertuples(index=False)} for fn, g in clean.groupby("filename")}
    for fi, fn in enumerate(files):
        d = by_file[fn]
        for wi in range(N_WINDOWS):
            end_sec = (wi + 1) * 5
            for t in [s.strip() for s in d.get(end_sec, "").split(";") if s.strip()]:
                if t in c2i:
                    y[fi, wi, c2i[t]] = 1.0

    sites_flat = np.repeat(np.array([parse_site(fn) for fn in files]), N_WINDOWS)
    hours_flat = np.repeat(np.array([parse_hour(fn) for fn in files], dtype=np.int64), N_WINDOWS)
    prior_tables = fit_simple_prior_tables(y.reshape(-1, y.shape[-1]), sites_flat, hours_flat)

    site_values = sorted(set(sites_flat.tolist()))
    site_to_idx = {s: i + 1 for i, s in enumerate(site_values)}
    return prior_tables, site_to_idx


def main() -> int:
    ap = argparse.ArgumentParser(description="Inference-only submission using exported fold models")
    ap.add_argument("--base-dir", default="/root/autodl-tmp/birdclef-2026")
    ap.add_argument("--data-2026", default="data/raw/birdclef-2026")
    ap.add_argument("--teacher-dir", default="models/perch_v2_cpu")
    ap.add_argument("--weights-dir", default="artifacts/exported_fold_models")
    ap.add_argument("--weights-prefix", default="student_cnn_2025_plus_2026_nodistill_keepperch")
    ap.add_argument(
        "--student-weight",
        default="artifacts/students/student_cnn_2025_plus_2026_nodistill_keepperch_ed28_cv45_c6_m45_ps0.40_seed42.pt",
        help="legacy single-student path; used as CNN fallback if --student-cnn-weight missing",
    )
    ap.add_argument("--student-blend", type=float, default=0.30, help="legacy single-student blend weight")
    ap.add_argument(
        "--student-cnn-weight",
        default="artifacts/students/student_cnn_2025_plus_2026_nodistill_keepperch_ed28_cv45_c6_m45_ps0.40_seed42.pt",
    )
    ap.add_argument(
        "--student-crnn-weight",
        default="artifacts/students/student_crnn_2025_plus_2026_nodistill_keepperch_ed28_cv45_c6_m45_ps0.40_seed42.pt",
    )
    ap.add_argument("--blend-perch", type=float, default=0.60)
    ap.add_argument("--blend-cnn", type=float, default=0.25)
    ap.add_argument("--blend-crnn", type=float, default=0.15)
    ap.add_argument("--proto-d-model", type=int, default=320)
    ap.add_argument("--prior-scale", type=float, default=0.40)
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    data_2026 = (base_dir / args.data_2026).resolve()
    teacher_dir = (base_dir / args.teacher_dir).resolve()
    weights_dir = (base_dir / args.weights_dir).resolve()
    student_weight = (base_dir / args.student_weight).resolve()
    student_cnn_weight = (base_dir / args.student_cnn_weight).resolve()
    student_crnn_weight = (base_dir / args.student_crnn_weight).resolve()

    teacher = PerchTeacher(
        teacher_dir=teacher_dir,
        taxonomy_csv=data_2026 / "taxonomy.csv",
        sample_submission_csv=data_2026 / "sample_submission.csv",
    )

    prior_tables, site_to_idx = build_training_priors(data_2026, teacher.primary_labels)
    student_cnn = None
    student_crnn = None
    legacy_student = None
    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=1024, win_length=1024, hop_length=320, n_mels=128, f_min=20, f_max=14000
    )
    if (not student_cnn_weight.exists()) and student_weight.exists():
        student_cnn_weight = student_weight

    if student_cnn_weight.exists():
        student_cnn = StudentCNN(emb_dim=1536, n_classes=len(teacher.primary_labels)).to("cpu")
        student_cnn.load_state_dict(torch.load(student_cnn_weight, map_location="cpu"))
        student_cnn.eval()
        print(f"use_student_cnn_logits={student_cnn_weight}")
    else:
        print(f"use_student_cnn_logits=NO (missing {student_cnn_weight})")

    if student_crnn_weight.exists():
        student_crnn = StudentCRNN(emb_dim=1536, n_classes=len(teacher.primary_labels)).to("cpu")
        student_crnn.load_state_dict(torch.load(student_crnn_weight, map_location="cpu"))
        student_crnn.eval()
        print(f"use_student_crnn_logits={student_crnn_weight}")
    else:
        print(f"use_student_crnn_logits=NO (missing {student_crnn_weight})")

    if (student_cnn is None) and (student_crnn is None) and student_weight.exists():
        legacy_student = StudentCNN(emb_dim=1536, n_classes=len(teacher.primary_labels)).to("cpu")
        legacy_student.load_state_dict(torch.load(student_weight, map_location="cpu"))
        legacy_student.eval()
        print(f"use_legacy_student_logits={student_weight} blend={args.student_blend:.2f}")

    test_dir = data_2026 / "test_soundscapes"
    if test_dir.exists() and any(test_dir.glob("*.ogg")):
        test_paths = sorted(test_dir.glob("*.ogg"))
    else:
        # dry-run fallback
        train_dir = data_2026 / "train_soundscapes"
        train_paths = sorted(train_dir.glob("*.ogg")) if train_dir.exists() else []
        if len(train_paths) == 0:
            train_paths = sorted(data_2026.glob("*.ogg"))
        test_paths = train_paths[:20]

    print(f"inference files={len(test_paths)}")

    all_scores = []
    row_ids = []

    n_sites = max(site_to_idx.values(), default=0) + 1
    fold_models = []
    for fold in [1, 2, 3, 4]:
        m = ProtoSSM(
            n_classes=len(teacher.primary_labels),
            d_model=args.proto_d_model,
            n_sites=n_sites,
        ).to("cpu")
        ckpt = weights_dir / f"{args.weights_prefix}_fold{fold}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"missing fold weight: {ckpt}")
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        m.eval()
        fold_models.append(m)

    for p in test_paths:
        windows = read_60s_windows(p)
        t_logits, t_emb = teacher.infer(windows)
        if (student_cnn is not None) or (student_crnn is not None):
            with torch.no_grad():
                w = torch.from_numpy(windows)
                mel = torch.log(mel_tf(w) + 1e-6).unsqueeze(1)
                logits_mix = []
                weight_mix = []

                logits_mix.append(t_logits.astype(np.float32))
                weight_mix.append(float(args.blend_perch))

                if student_cnn is not None:
                    _, cnn_logits = student_cnn(mel)
                    logits_mix.append(cnn_logits.numpy().astype(np.float32))
                    weight_mix.append(float(args.blend_cnn))

                if student_crnn is not None:
                    _, crnn_logits = student_crnn(mel)
                    logits_mix.append(crnn_logits.numpy().astype(np.float32))
                    weight_mix.append(float(args.blend_crnn))

                wsum = float(sum(weight_mix))
                if wsum <= 0:
                    t_logits = t_logits.astype(np.float32)
                else:
                    merged = np.zeros_like(t_logits, dtype=np.float32)
                    for arr_logits, wv in zip(logits_mix, weight_mix):
                        merged += (float(wv) / wsum) * arr_logits
                    t_logits = merged
        elif legacy_student is not None:
            # Legacy single-student behavior
            with torch.no_grad():
                w = torch.from_numpy(windows)
                mel = torch.log(mel_tf(w) + 1e-6).unsqueeze(1)
                _, stu_logits = legacy_student(mel)
                s_logits = stu_logits.numpy().astype(np.float32)
            t_logits = ((1.0 - args.student_blend) * t_logits + args.student_blend * s_logits).astype(np.float32)

        site = parse_site(p.name)
        hour = parse_hour(p.name)
        sites_flat = np.array([site] * N_WINDOWS)
        hours_flat = np.array([hour] * N_WINDOWS, dtype=np.int64)
        prior = prior_logits_from_tables(sites_flat, hours_flat, prior_tables)
        t_logits_use = t_logits + args.prior_scale * prior

        x = torch.tensor(t_emb[None, :, :], dtype=torch.float32)
        l = torch.tensor(t_logits_use[None, :, :], dtype=torch.float32)
        sid = torch.tensor([site_to_idx.get(site, 0)], dtype=torch.long)
        hid = torch.tensor([hour], dtype=torch.long)

        fold_probs = []
        with torch.no_grad():
            for m in fold_models:
                out = m(x, l, sid, hid)
                fold_probs.append(torch.sigmoid(out).cpu().numpy()[0])
        probs = np.mean(fold_probs, axis=0)
        probs = postprocess_probs_filewise(probs, n_windows=N_WINDOWS)

        stem = p.stem
        for i in range(N_WINDOWS):
            row_ids.append(f"{stem}_{(i+1)*5}")
        all_scores.append(probs)

    scores = np.concatenate(all_scores, axis=0).astype(np.float32)
    sub = pd.DataFrame(scores, columns=teacher.primary_labels)
    sub.insert(0, "row_id", row_ids)
    out_path = Path(args.out).resolve()
    sub.to_csv(out_path, index=False)
    print(f"saved={out_path} shape={sub.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
