"""
ANN-to-SNN conversion pipeline.

Steps
-----
1. Load the trained DQN and calibration statistics.
2. Compute layer thresholds (via normalize.py).
3. Scale each layer's weights:
       W_snn[i] = W_ann[i] * (λ_{i-1} / λ_i)
   where λ_i is the threshold of layer i and λ_0 = 1.0 (input is [0,1]).
   This ensures that a unit of activation in the ANN corresponds to a
   unit of spike rate in the SNN.
4. Wrap the scaled layers in SNNFromDQN.
5. Verify action agreement on a set of recorded frames.
6. Save the SNN to disk.

Theory
------
The ANN ReLU output at layer i, call it a_i, is approximated by a neuron
that fires at rate  r_i = a_i / λ_i  spikes per timestep.
The downstream layer i+1 receives current  W_{i+1} * r_i * λ_i  per timestep,
which equals  W_{i+1} * a_i  — the same as the ANN.
So by scaling  W_snn_{i+1} = W_ann_{i+1} * λ_i / λ_{i+1},  the SNN
correctly reproduces the ANN's forward pass at infinite T.

Usage
-----
# Full pipeline (calibrate → convert → verify → save)
python conversion/convert.py --checkpoint checkpoints/best.pt

# Just convert (if calibration_stats.pkl already exists)
python conversion/convert.py --checkpoint checkpoints/best.pt --no-calibrate

# Convert and verify with minimum agreement threshold
python conversion/convert.py --checkpoint checkpoints/best.pt --min-agreement 0.90
"""

from __future__ import annotations

import argparse
import copy
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import (
    CALIBRATION_EPISODES,
    CHECKPOINT_DIR,
    FRAME_STACK,
    IMG_H,
    IMG_W,
    PERCENTILE,
    T_VALUES,
)
from conversion.calibrate import calibrate
from conversion.normalize import compute_thresholds
from models.ann import DQN
from models.snn import SNNFromDQN


# ── Weight scaling ────────────────────────────────────────────────────────────

def scale_layer(
    src: nn.Module,
    scale: float,
    new_cls=None,
    **kwargs,
) -> nn.Module:
    """
    Create a new layer of the same type as src with weights scaled by `scale`.
    """
    if new_cls is None:
        new_cls = type(src)

    dst = copy.deepcopy(src)
    with torch.no_grad():
        dst.weight.mul_(scale)
        if dst.bias is not None:
            dst.bias.mul_(scale)
    return dst


def build_snn_from_ann(
    ann:        DQN,
    thresholds: list[float],
) -> SNNFromDQN:
    """
    Extract layers from the ANN, scale their weights, and wrap in SNNFromDQN.

    Parameters
    ----------
    ann : DQN
        Trained ANN (on CPU is fine for conversion).
    thresholds : list[float]
        [λ_conv1, λ_conv2, λ_conv3, λ_fc1]  — in layer order.

    Returns
    -------
    SNNFromDQN ready to run inference.
    """
    assert len(thresholds) == 4, f"Expected 4 thresholds, got {len(thresholds)}"

    # Layer order in ann.conv: Conv ReLU Conv ReLU Conv ReLU
    # Indices 0, 2, 4 are the Conv2d modules; 1, 3, 5 are ReLUs
    ann_convs = [ann.conv[0], ann.conv[2], ann.conv[4]]

    # Layer order in ann.fc: Linear ReLU Linear
    # Index 0 is hidden fc, index 2 is output fc
    ann_fcs = [ann.fc[0], ann.fc[2]]

    # Pre-layer scales: λ_0 = 1.0 (image input), then λ_1, λ_2, λ_3
    pre_scales  = [1.0] + thresholds[:3]   # input → conv1 → conv2 → conv3
    post_scales = thresholds               # conv1, conv2, conv3, fc1

    scaled_convs = []
    for i, conv in enumerate(ann_convs):
        scale = pre_scales[i] / post_scales[i]
        scaled_convs.append(scale_layer(conv, scale))

    # Hidden FC: pre-scale = λ_conv3, post-scale = λ_fc1
    fc_hidden_scale = thresholds[2] / thresholds[3]
    scaled_fc_hidden = scale_layer(ann_fcs[0], fc_hidden_scale)

    # Output FC: no IF neuron, so scale by 1.0 / λ_fc1 to normalise
    # the membrane accumulation back to ANN Q-value magnitude
    output_scale = 1.0 / thresholds[3]
    scaled_fc_out = scale_layer(ann_fcs[1], output_scale)

    snn = SNNFromDQN(
        conv_layers=scaled_convs,
        fc_layers=[scaled_fc_hidden, scaled_fc_out],
        thresholds=thresholds,
    )
    return snn


# ── Verification ──────────────────────────────────────────────────────────────

def verify_action_agreement(
    ann:         DQN,
    snn:         SNNFromDQN,
    n_episodes:  int   = 50,
    T:           int   = 64,
    device:      torch.device | None = None,
    min_agreement: float = 0.90,
) -> float:
    """
    Compare ANN vs SNN action selection on live gameplay frames.

    Returns
    -------
    float — fraction of frames where both nets pick the same action.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ann.to(device).eval()
    snn.to(device).eval()

    from envs.doom_env import DoomEnv
    env = DoomEnv()

    agree = 0
    total = 0

    for _ in range(n_episodes):
        obs  = env.reset()
        done = False
        while not done:
            obs_t = torch.as_tensor(
                obs, dtype=torch.float32, device=device
            ).unsqueeze(0)
            with torch.no_grad():
                ann_action = int(ann(obs_t).argmax(dim=1).item())
                snn_action = int(snn(obs_t, T=T).argmax(dim=1).item())
            agree += int(ann_action == snn_action)
            total += 1
            _, _, done = env.step(ann_action)   # follow ANN policy

    env.close()
    agreement = agree / max(total, 1)
    status    = "✓" if agreement >= min_agreement else "✗"
    print(f"\nAction agreement @ T={T}: {agreement*100:.1f}%  "
          f"({agree}/{total} frames)  {status}")
    if agreement < min_agreement:
        print(f"  [!] Below target {min_agreement*100:.0f}%. "
              f"Try lowering --percentile or increasing T.")
    return agreement


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert trained DQN to SNN")
    p.add_argument("--checkpoint",    type=str,
                   default=f"{CHECKPOINT_DIR}/best.pt")
    p.add_argument("--stats",         type=str,
                   default="calibration_stats.pkl")
    p.add_argument("--out",           type=str,
                   default=f"{CHECKPOINT_DIR}/best_snn.pt")
    p.add_argument("--no-calibrate",  action="store_true",
                   help="Skip calibration (use existing stats file)")
    p.add_argument("--percentile",    type=float, default=PERCENTILE)
    p.add_argument("--verify",        action="store_true", default=True)
    p.add_argument("--no-verify",     action="store_false", dest="verify")
    p.add_argument("--verify-T",      type=int, default=64)
    p.add_argument("--verify-eps",    type=int, default=50)
    p.add_argument("--min-agreement", type=float, default=0.90)
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Step 1: Calibrate ─────────────────────────────────────────────────────
    if not args.no_calibrate:
        print("═" * 60)
        print("Step 1: Calibrating activations …")
        print("═" * 60)
        calibrate(
            checkpoint=args.checkpoint,
            n_episodes=CALIBRATION_EPISODES,
            out_path=args.stats,
            device=device,
        )
    else:
        print(f"Skipping calibration — using existing stats: {args.stats}")

    # ── Step 2: Compute thresholds ────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("Step 2: Computing thresholds …")
    print("═" * 60)
    thresh_dict = compute_thresholds(args.stats, args.percentile)
    # Ordered list: relu_0 (conv1), relu_1 (conv2), relu_2 (conv3), relu_3 (fc1)
    thresholds  = [thresh_dict[k] for k in sorted(thresh_dict.keys())]

    # ── Step 3: Load ANN and build SNN ───────────────────────────────────────
    print("\n" + "═" * 60)
    print("Step 3: Converting ANN → SNN …")
    print("═" * 60)
    ann  = DQN()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ann.load_state_dict(ckpt.get("online_net", ckpt))
    ann.eval()

    snn = build_snn_from_ann(ann, thresholds)
    print("SNN built successfully.")
    total_params = sum(p.numel() for p in snn.parameters())
    print(f"SNN parameters: {total_params:,}")

    # ── Step 4: Verify ────────────────────────────────────────────────────────
    if args.verify:
        print("\n" + "═" * 60)
        print(f"Step 4: Verifying action agreement (T={args.verify_T}) …")
        print("═" * 60)
        agreement = verify_action_agreement(
            ann, snn,
            n_episodes   = args.verify_eps,
            T            = args.verify_T,
            device       = device,
            min_agreement= args.min_agreement,
        )

    # ── Step 5: Save ──────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("Step 5: Saving SNN …")
    print("═" * 60)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(snn, args.out)
    print(f"SNN saved to: {args.out}")
    print("\nConversion complete ✓")


if __name__ == "__main__":
    main()
