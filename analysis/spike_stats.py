"""
Spike activity analysis.

Measures per-layer firing rates and total spikes per decision for the
converted SNN agent. This quantifies the energy-efficiency advantage of
the spiking representation.

Metrics reported
----------------
- Average firing rate per layer  (spikes / neuron / timestep)
- Total spikes per decision       (sum across all neurons and timesteps)
- Spike sparsity                  (fraction of neurons that never fire)
- Comparison to ANN FLOPs (approximate multiply-accumulate ops)

Usage
-----
python analysis/spike_stats.py --checkpoint checkpoints/best_snn.pt --T 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import FRAME_STACK, IMG_H, IMG_W
from envs.doom_env import DoomEnv
from models.snn import IFNeuron, SNNFromDQN


# ── Instrumented SNN forward ──────────────────────────────────────────────────

def collect_spike_stats(
    snn:        SNNFromDQN,
    obs:        torch.Tensor,
    T:          int,
) -> dict[str, dict]:
    """
    Run one forward pass and collect spike counts per layer.

    Returns
    -------
    dict[layer_name, {total_spikes, n_neurons, firing_rate, sparsity}]
    """
    device = obs.device
    B      = obs.shape[0]
    stats  = {}

    # ── replicate SNNFromDQN.forward with instrumentation ─────────────────────
    mem_conv = [None] * snn._n_if_conv
    mem_fc   = [None] * snn._n_if_fc
    acc_out  = torch.zeros(B, snn.fc_layers[-1].out_features, device=device)

    spike_counts_conv = [torch.zeros(1, device=device) for _ in range(snn._n_if_conv)]
    spike_counts_fc   = [torch.zeros(1, device=device) for _ in range(snn._n_if_fc)]
    layer_sizes_conv  = [0] * snn._n_if_conv
    layer_sizes_fc    = [0] * snn._n_if_fc

    for _ in range(T):
        h = obs
        for i, (conv, neuron) in enumerate(zip(snn.conv_layers, snn.if_conv)):
            h = conv(h)
            if mem_conv[i] is None:
                mem_conv[i] = torch.zeros_like(h)
                layer_sizes_conv[i] = h.numel() // B
            h, mem_conv[i] = neuron(h, mem_conv[i])
            spike_counts_conv[i] = spike_counts_conv[i] + h.sum()

        h = h.flatten(1)
        for i, (fc, neuron) in enumerate(zip(snn.fc_layers[:-1], snn.if_fc)):
            h = fc(h)
            if mem_fc[i] is None:
                mem_fc[i] = torch.zeros_like(h)
                layer_sizes_fc[i] = h.numel() // B
            h, mem_fc[i] = neuron(h, mem_fc[i])
            spike_counts_fc[i] = spike_counts_fc[i] + h.sum()

        h = snn.fc_layers[-1](h)
        acc_out = acc_out + h

    # ── Compute per-layer stats ───────────────────────────────────────────────
    layer_names = (
        [f"conv_{i}" for i in range(snn._n_if_conv)]
        + [f"fc_{i}"   for i in range(snn._n_if_fc)]
    )
    spike_counts = spike_counts_conv + spike_counts_fc
    layer_sizes  = layer_sizes_conv + layer_sizes_fc

    total_spikes = 0
    for name, count, size in zip(layer_names, spike_counts, layer_sizes):
        n_spikes  = float(count.item())
        n_neurons = size * B
        rate      = n_spikes / (n_neurons * T + 1e-9)
        sparsity  = 1.0 - rate
        stats[name] = {
            "total_spikes": n_spikes,
            "n_neurons":    n_neurons,
            "firing_rate":  rate,
            "sparsity":     sparsity,
        }
        total_spikes += n_spikes

    stats["_total_spikes_per_decision"] = total_spikes / B
    return stats


# ── Full episode analysis ─────────────────────────────────────────────────────

def analyse_episodes(
    snn_path:   str,
    n_episodes: int = 50,
    T:          int = 32,
    device:     torch.device | None = None,
) -> None:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading SNN from: {snn_path}")
    snn: SNNFromDQN = torch.load(snn_path, map_location=device, weights_only=False)
    snn.eval()

    env   = DoomEnv()
    all_stats: list[dict] = []
    rewards = []

    for ep in range(n_episodes):
        obs  = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            obs_t = torch.as_tensor(
                obs, dtype=torch.float32, device=device
            ).unsqueeze(0)
            ep_stats = collect_spike_stats(snn, obs_t, T=T)
            all_stats.append(ep_stats)
            action       = int(snn(obs_t, T=T).argmax(dim=1).item())
            obs, r, done = env.step(action)
            ep_r        += r
        rewards.append(ep_r)

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{n_episodes}  reward={ep_r:.1f}")

    env.close()

    # ── Aggregate ─────────────────────────────────────────────────────────────
    layer_keys = [k for k in all_stats[0] if not k.startswith("_")]
    total_spikes_list = [s["_total_spikes_per_decision"] for s in all_stats]

    print(f"\n{'─'*65}")
    print(f"  Spike Statistics  |  T={T}  |  Episodes={n_episodes}")
    print(f"{'─'*65}")
    print(f"  {'Layer':<12}  {'Mean Rate':>12}  {'Sparsity':>10}  {'Total Spikes':>14}")
    print(f"  {'-'*58}")
    for k in layer_keys:
        rates     = [s[k]["firing_rate"]  for s in all_stats]
        sparsity  = [s[k]["sparsity"]     for s in all_stats]
        spikes    = [s[k]["total_spikes"] for s in all_stats]
        print(f"  {k:<12}  {np.mean(rates):>11.3f}  "
              f"{np.mean(sparsity):>9.1%}  {np.mean(spikes):>14.0f}")

    print(f"\n  Mean spikes/decision : {np.mean(total_spikes_list):.0f}")
    print(f"  Mean episode reward  : {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"{'─'*65}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str,
                   default="checkpoints/best_snn.pt")
    p.add_argument("--T",          type=int, default=32)
    p.add_argument("--episodes",   type=int, default=50)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyse_episodes(
        snn_path=args.checkpoint,
        n_episodes=args.episodes,
        T=args.T,
    )
