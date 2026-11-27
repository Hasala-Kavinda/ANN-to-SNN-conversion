"""
Per-layer ANN-vs-SNN conversion diagnostics.

Compares ANN ReLU activations `a_i` against scaled SNN firing rates
`rate_i * lambda_i` on calibration frames to pinpoint which layer saturates
or under-fires at a given T.

Usage
-----
python analysis/conversion_diagnostics.py \
    --ann-checkpoint checkpoints/best.pt \
    --snn-checkpoint checkpoints/best_snn.pt \
    --frames calibration_frames.npz \
    --T 32
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import FRAME_STACK
from models.ann import DQN
from models.snn import SNNFromDQN


def compute_layer_diagnostics(
    ann:         DQN,
    snn:         SNNFromDQN,
    frames_u8:   np.ndarray,
    T:           int,
    raw_thresholds: list[float] | None = None,
    device:      torch.device | None = None,
    max_frames:  int = 256,
) -> dict[str, dict]:
    """
    Compare ANN ReLU activations against SNN spike rates on `max_frames` frames.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ann.to(device).eval()
    snn.to(device).eval()

    xb = torch.from_numpy(frames_u8[:max_frames]).float().div_(255.0).to(device)

    # Collect ANN ReLU activations
    ann_acts: list[np.ndarray] = []
    hooks = []
    for m in ann.modules():
        if isinstance(m, torch.nn.ReLU):
            hooks.append(
                m.register_forward_hook(
                    lambda _m, _i, out: ann_acts.append(out.detach().cpu().numpy().ravel())
                )
            )
    with torch.no_grad():
        _ = ann(xb)
    for h in hooks:
        h.remove()

    # Collect SNN per-neuron firing rates across the same batch
    B = xb.shape[0]
    ratio = snn.mem_init_ratio
    mem_conv = [None] * snn._n_if_conv
    mem_fc   = [None] * snn._n_if_fc
    spk_conv = [None] * snn._n_if_conv
    spk_fc   = [None] * snn._n_if_fc

    with torch.no_grad():
        cur0 = snn.conv_layers[0](xb)
        for _ in range(T):
            h = cur0
            for i, (conv, neuron) in enumerate(zip(snn.conv_layers, snn.if_conv)):
                if i > 0:
                    h = conv(h)
                if mem_conv[i] is None:
                    mem_conv[i] = snn._init_mem(h, neuron.threshold, ratio)
                    spk_conv[i] = torch.zeros_like(h)
                h, mem_conv[i] = neuron(h, mem_conv[i])
                spk_conv[i] = spk_conv[i] + h
            h = h.flatten(1)
            for i, (fc, neuron) in enumerate(zip(snn.fc_layers[:-1], snn.if_fc)):
                h = fc(h)
                if mem_fc[i] is None:
                    mem_fc[i] = snn._init_mem(h, neuron.threshold, ratio)
                    spk_fc[i] = torch.zeros_like(h)
                h, mem_fc[i] = neuron(h, mem_fc[i])
                spk_fc[i] = spk_fc[i] + h

    layer_names = [f"conv_{i}" for i in range(snn._n_if_conv)] + [f"fc_{i}" for i in range(snn._n_if_fc)]
    snn_rates = [s.cpu().numpy().ravel() / float(T) for s in (spk_conv + spk_fc)]
    lambdas = raw_thresholds if raw_thresholds is not None else [1.0] * len(layer_names)

    rng = np.random.default_rng(0)
    diag: dict[str, dict] = {}
    for name, a_ann, r_snn, lam in zip(layer_names, ann_acts, snn_rates, lambdas):
        a_snn_est = r_snn * lam
        corr = float(np.corrcoef(a_ann, a_snn_est)[0, 1]) if np.std(a_ann) > 0 and np.std(a_snn_est) > 0 else 0.0
        clip_frac = float(np.mean(a_ann > lam)) if raw_thresholds is not None else 0.0
        dead_frac = float(np.mean(r_snn == 0.0))

        # Subsample points for plotting
        if len(a_ann) > 4000:
            idx = rng.choice(len(a_ann), size=4000, replace=False)
            x_pts, y_pts = a_ann[idx], a_snn_est[idx]
        else:
            x_pts, y_pts = a_ann, a_snn_est

        diag[name] = {
            "pearson_r":  corr,
            "clip_frac":  clip_frac,
            "dead_frac":  dead_frac,
            "lambda":     float(lam),
            "ann_sample": x_pts,
            "snn_sample": y_pts,
        }
    return diag


def plot_diagnostics(diag: dict[str, dict], T: int, out_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.5))
    for ax, (name, d) in zip(axes.ravel(), diag.items()):
        x = d["ann_sample"]
        y = d["snn_sample"]
        ax.scatter(x, y, s=4, alpha=0.25, color="#4c72b0")
        lim = max(float(x.max()), float(y.max()), 1e-3)
        ax.plot([0, lim], [0, lim], "r--", lw=1.2, label="ideal y=x")
        ax.set_title(
            f"{name} | r={d['pearson_r']:.4f} | clip={d['clip_frac']*100:.2f}% | dead={d['dead_frac']*100:.1f}%",
            fontsize=9.5,
        )
        ax.set_xlabel("ANN ReLU activation")
        ax.set_ylabel("SNN rate × λ")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Per-Layer ANN vs SNN Conversion Diagnostics (T={T})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ann-checkpoint", type=str, default="checkpoints/best.pt")
    p.add_argument("--snn-checkpoint", type=str, default="checkpoints/best_snn.pt")
    p.add_argument("--frames",         type=str, default="calibration_frames.npz")
    p.add_argument("--T",              type=int, default=32)
    p.add_argument("--out",            type=str, default="results/conversion_diagnostics.png")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ann_checkpoint, map_location="cpu")
    sd = ckpt.get("online_net", ckpt)
    in_ch = int(sd["conv.0.weight"].shape[1]) if "conv.0.weight" in sd else FRAME_STACK
    ann = DQN(in_channels=in_ch)
    ann.load_state_dict(sd)

    snn_payload = torch.load(args.snn_checkpoint, map_location="cpu", weights_only=False)
    raw_thresholds = None
    if isinstance(snn_payload, dict):
        raw_thresholds = snn_payload.get("metadata", {}).get("raw_thresholds")
    snn = SNNFromDQN.load_checkpoint(args.snn_checkpoint, device=device)

    frames_u8 = np.load(args.frames)["frames"]
    diag = compute_layer_diagnostics(
        ann, snn, frames_u8, T=args.T, raw_thresholds=raw_thresholds, device=device
    )
    os.makedirs(Path(args.out).parent, exist_ok=True)
    plot_diagnostics(diag, T=args.T, out_path=args.out)


if __name__ == "__main__":
    main()
