"""
Calibration: run the trained DQN on recorded game frames and collect
per-layer activation statistics needed for threshold normalization.

How it works
------------
1. Load the trained DQN checkpoint.
2. Play CALIBRATION_EPISODES episodes under the greedy policy.
3. Use PyTorch forward hooks to capture the output of each ReLU layer.
4. Save the raw activation tensors (flattened) to calibration_stats.pkl.

The saved file is consumed by normalize.py and convert.py.

Usage
-----
python conversion/calibrate.py --checkpoint checkpoints/best.pt
python conversion/calibrate.py --checkpoint checkpoints/best.pt --episodes 200
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import CALIBRATION_EPISODES, CHECKPOINT_DIR
from envs.doom_env import DoomEnv
from models.ann import DQN


def calibrate(
    checkpoint:  str,
    n_episodes:  int   = CALIBRATION_EPISODES,
    out_path:    str   = "calibration_stats.pkl",
    device:      torch.device | None = None,
) -> dict[str, np.ndarray]:
    """
    Run the DQN greedily for n_episodes and collect per-layer activations.

    Returns
    -------
    dict mapping layer_name -> np.ndarray of all observed activations
    (flattened across batch and spatial dims).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {checkpoint}")
    ann  = DQN().to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    sd   = ckpt.get("online_net", ckpt)
    ann.load_state_dict(sd)
    ann.eval()

    # ── Register hooks on every ReLU ─────────────────────────────────────────
    # Layer order: conv_relu_0, conv_relu_1, conv_relu_2, fc_relu_0
    activation_store: dict[str, list[np.ndarray]] = {}
    hooks = []

    relu_idx = [0]

    def make_hook(name: str):
        activation_store[name] = []
        def hook(module, inp, out):
            activation_store[name].append(
                out.detach().cpu().numpy().flatten()
            )
        return hook

    for module in ann.modules():
        if isinstance(module, torch.nn.ReLU):
            name = f"relu_{relu_idx[0]}"
            hooks.append(module.register_forward_hook(make_hook(name)))
            relu_idx[0] += 1

    print(f"Registered hooks on {relu_idx[0]} ReLU layers: "
          f"{list(activation_store.keys())}")

    # ── Run episodes ──────────────────────────────────────────────────────────
    env    = DoomEnv()
    frames = 0

    for ep in range(n_episodes):
        obs  = env.reset()
        done = False
        while not done:
            obs_t = torch.as_tensor(
                obs, dtype=torch.float32, device=device
            ).unsqueeze(0)
            with torch.no_grad():
                ann(obs_t)   # hooks collect activations as side-effect
            action       = int(ann(obs_t).argmax(dim=1).item())
            obs, _, done = env.step(action)
            frames      += 1

        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1}/{n_episodes}  frames so far: {frames:,}")

    env.close()
    for h in hooks:
        h.remove()

    print(f"\nTotal frames collected: {frames:,}")

    # ── Concatenate and save ──────────────────────────────────────────────────
    stats: dict[str, np.ndarray] = {}
    for name, arrays in activation_store.items():
        combined   = np.concatenate(arrays)
        stats[name] = combined
        print(f"  {name}: {len(combined):,} values  "
              f"max={combined.max():.4f}  "
              f"99.9th={np.percentile(combined, 99.9):.4f}")

    with open(out_path, "wb") as f:
        pickle.dump(stats, f)

    print(f"\nCalibration stats saved to: {out_path}")
    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str,
                   default=f"{CHECKPOINT_DIR}/best.pt")
    p.add_argument("--episodes",   type=int, default=CALIBRATION_EPISODES)
    p.add_argument("--out",        type=str, default="calibration_stats.pkl")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    calibrate(
        checkpoint=args.checkpoint,
        n_episodes=args.episodes,
        out_path=args.out,
    )
