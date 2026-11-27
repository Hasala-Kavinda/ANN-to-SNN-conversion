"""
Spike activity analysis.

Measures per-layer firing rates, temporal/neuron sparsity, total spikes per
decision, and analytical SNN SOPs vs ANN MACs for the converted SNN agent.

Note on energy accounting:
  Conv1 receives constant-current analog pixel inputs (MAC-driven), whereas
  Conv2, Conv3, FC1, and FC2 (output readout) are spike-driven (AC/SOP-driven).

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import IMG_H, IMG_W
from envs.doom_env import DoomEnv
from models.snn import SNNFromDQN


# ── Instrumented SNN forward ──────────────────────────────────────────────────

def collect_spike_stats_and_q(
    snn: SNNFromDQN,
    obs: torch.Tensor,
    T:   int,
) -> tuple[torch.Tensor, dict[str, dict]]:
    """Run one T-step forward pass and return (q_values, per_layer_stats)."""
    return snn.forward_with_stats(obs, T=T)


def collect_spike_stats(
    snn: SNNFromDQN,
    obs: torch.Tensor,
    T:   int,
) -> dict[str, dict]:
    """Run one forward pass and collect spike counts per layer."""
    _, stats = snn.forward_with_stats(obs, T=T)
    return stats


# ── Analytical ANN MACs vs SNN SOPs ───────────────────────────────────────────

def compute_ann_macs_and_snn_sops(
    snn:          SNNFromDQN,
    layer_spikes: dict[str, float],
    img_h:        int = IMG_H,
    img_w:        int = IMG_W,
) -> dict[str, float]:
    """
    Compute per-decision ANN MACs and SNN Synaptic Operations (SOPs).

    Parameters
    ----------
    snn : SNNFromDQN
    layer_spikes : dict[str, float]
        Mean spikes per decision for each IF layer ('conv_0', 'conv_1', 'conv_2', 'fc_0').
    """
    h, w = img_h, img_w
    conv_macs: list[float] = []
    conv_shapes: list[tuple[int, int]] = []

    for conv in snn.conv_layers:
        kh, kw = conv.kernel_size if isinstance(conv.kernel_size, tuple) else (conv.kernel_size, conv.kernel_size)
        sh, sw = conv.stride if isinstance(conv.stride, tuple) else (conv.stride, conv.stride)
        ph, pw = conv.padding if isinstance(conv.padding, tuple) else (conv.padding, conv.padding)
        h_out = (h + 2 * ph - kh) // sh + 1
        w_out = (w + 2 * pw - kw) // sw + 1
        macs = float(h_out * w_out * conv.out_channels * (conv.in_channels // conv.groups) * kh * kw)
        conv_macs.append(macs)
        conv_shapes.append((h, w))
        h, w = h_out, w_out

    fc_macs: list[float] = [float(fc.in_features * fc.out_features) for fc in snn.fc_layers]

    # Conv0 is analog direct-current input (MAC-driven, not spike-driven)
    conv0_analog_macs = conv_macs[0]

    # Spike-driven layers: conv_1, conv_2, fc_0 (hidden), fc_1 (readout)
    sops = 0.0
    for idx in range(1, len(snn.conv_layers)):
        prev_name = f"conv_{idx - 1}"
        h_in, w_in = conv_shapes[idx]
        c_next = snn.conv_layers[idx]
        kh, kw = c_next.kernel_size if isinstance(c_next.kernel_size, tuple) else (c_next.kernel_size, c_next.kernel_size)
        sh, sw = c_next.stride if isinstance(c_next.stride, tuple) else (c_next.stride, c_next.stride)
        ph, pw = c_next.padding if isinstance(c_next.padding, tuple) else (c_next.padding, c_next.padding)
        h_out = (h_in + 2 * ph - kh) // sh + 1
        w_out = (w_in + 2 * pw - kw) // sw + 1
        fan_out = (h_out * w_out * c_next.out_channels * kh * kw) / max(h_in * w_in * c_next.in_channels, 1)
        sops += layer_spikes.get(prev_name, 0.0) * fan_out

    # conv_2 -> fc_0
    last_conv_name = f"conv_{len(snn.conv_layers) - 1}"
    sops += layer_spikes.get(last_conv_name, 0.0) * float(snn.fc_layers[0].out_features)

    # fc_0 -> fc_1 (readout)
    sops += layer_spikes.get("fc_0", 0.0) * float(snn.fc_layers[1].out_features)

    ann_macs_spiking = sum(conv_macs[1:]) + sum(fc_macs)
    ann_macs_total   = sum(conv_macs) + sum(fc_macs)

    return {
        "ann_macs_total":          ann_macs_total,
        "ann_macs_spiking_layers": ann_macs_spiking,
        "conv0_analog_macs":       conv0_analog_macs,
        "snn_sops_spiking_layers": float(sops),
        "sop_to_mac_ratio":        float(sops / max(ann_macs_spiking, 1.0)),
    }


# ── Full episode analysis ─────────────────────────────────────────────────────

def analyse_episodes(
    snn_path:   str,
    n_episodes: int = 50,
    T:          int = 32,
    device:     torch.device | None = None,
    seed:       int = 0,
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading SNN from: {snn_path}")
    snn = SNNFromDQN.load_checkpoint(snn_path, device=device)

    env = DoomEnv(seed=seed)
    all_stats: list[dict] = []
    rewards: list[float]  = []

    for ep in range(n_episodes):
        obs  = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            obs_t = torch.as_tensor(
                obs, dtype=torch.float32, device=device
            ).unsqueeze(0)
            with torch.no_grad():
                q_vals, ep_stats = collect_spike_stats_and_q(snn, obs_t, T=T)
            all_stats.append(ep_stats)
            action       = int(q_vals.argmax(dim=1).item())
            obs, r, done = env.step(action)
            ep_r        += r
        rewards.append(ep_r)

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{n_episodes}  reward={ep_r:.1f}")

    env.close()

    # ── Aggregate ─────────────────────────────────────────────────────────────
    layer_keys = [k for k in all_stats[0] if not k.startswith("_")]
    total_spikes_list = [s["_total_spikes_per_decision"] for s in all_stats]

    layer_summary: dict[str, dict[str, float]] = {}
    mean_spikes_per_layer: dict[str, float] = {}

    print(f"\n{'─'*78}")
    print(f"  Spike Statistics  |  T={T}  |  Episodes={n_episodes}")
    print(f"{'─'*78}")
    print(f"  {'Layer':<10}  {'Mean Rate':>10}  {'Temp Sparsity':>14}  {'Silent Neurons':>15}  {'Mean Spikes':>12}")
    print(f"  {'-'*70}")
    for k in layer_keys:
        rates       = [s[k]["firing_rate"]        for s in all_stats]
        t_sparsity  = [s[k]["temporal_sparsity"]  for s in all_stats]
        silent_frac = [s[k]["silent_neuron_frac"] for s in all_stats]
        spikes      = [s[k]["total_spikes"]       for s in all_stats]

        m_rate   = float(np.mean(rates))
        m_tsp    = float(np.mean(t_sparsity))
        m_silent = float(np.mean(silent_frac))
        m_spikes = float(np.mean(spikes))

        layer_summary[k] = {
            "firing_rate":        m_rate,
            "temporal_sparsity":  m_tsp,
            "silent_neuron_frac": m_silent,
            "mean_spikes":        m_spikes,
        }
        mean_spikes_per_layer[k] = m_spikes

        print(f"  {k:<10}  {m_rate:>10.3f}  {m_tsp:>13.1%}  {m_silent:>14.1%}  {m_spikes:>12.0f}")

    ops = compute_ann_macs_and_snn_sops(snn, mean_spikes_per_layer)
    mean_spikes_decision = float(np.mean(total_spikes_list))
    mean_reward          = float(np.mean(rewards))
    std_reward           = float(np.std(rewards))

    print(f"\n  Mean spikes/decision           : {mean_spikes_decision:,.0f}")
    print(f"  Conv1 analog MACs (1x/decision): {ops['conv0_analog_macs']:,.0f}  (direct-current input)")
    print(f"  ANN MACs (spike-driven layers) : {ops['ann_macs_spiking_layers']:,.0f}")
    print(f"  SNN SOPs (spike-driven layers) : {ops['snn_sops_spiking_layers']:,.0f}  "
          f"({ops['sop_to_mac_ratio']:.2f}x of ANN MACs)")
    print(f"  Mean episode reward            : {mean_reward:.1f} ± {std_reward:.1f}")
    print(f"{'─'*78}\n")

    return {
        "T":                        T,
        "n_episodes":               n_episodes,
        "mean_reward":              mean_reward,
        "std_reward":               std_reward,
        "mean_spikes_per_decision": mean_spikes_decision,
        "layers":                   layer_summary,
        "ops":                      ops,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_snn.pt")
    p.add_argument("--T",          type=int, default=32)
    p.add_argument("--episodes",   type=int, default=50)
    p.add_argument("--seed",       type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyse_episodes(
        snn_path=args.checkpoint,
        n_episodes=args.episodes,
        T=args.T,
        seed=args.seed,
    )
