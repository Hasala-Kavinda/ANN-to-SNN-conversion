"""
Generate all result plots and summary JSON for the report.

Plots produced
--------------
1. results_reward_vs_T.png     — SNN reward vs T, with ANN, Scripted, and Random baselines.
2. results_agreement_vs_T.png  — Action agreement (%) vs T.
3. results_spike_rates.png     — Per-layer spike firing rate bar chart at selected T.
4. results_summary.json        — Structured metrics for report tables.

Usage
-----
python analysis/plot_results.py \
    --ann-checkpoint  checkpoints/best.pt \
    --snn-checkpoint  checkpoints/best_snn.pt \
    --T-values 8 16 32 64 128 \
    --episodes 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import T_VALUES
from evaluation.eval_agent import (
    ANNAgent,
    RandomAgent,
    ScriptedAgent,
    SNNAgent,
    run_evaluation,
)


# ── Plot helpers ──────────────────────────────────────────────────────────────

STYLE = {
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "font.size": 11,
    "lines.linewidth": 2,
}

plt.rcParams.update(STYLE)


def plot_reward_vs_T(
    ann_reward: float,
    random_reward: float,
    T_list: list[int],
    snn_rewards: list[float],
    snn_stds: list[float],
    out_path: str = "results/results_reward_vs_T.png",
    scripted_reward: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))

    ax.errorbar(
        T_list,
        snn_rewards,
        yerr=snn_stds,
        marker="o",
        label="SNN",
        color="#e07b39",
        capsize=4,
    )
    ax.axhline(ann_reward, ls="--", color="#4c72b0", label=f"ANN ({ann_reward:.1f})")
    if scripted_reward is not None:
        ax.axhline(scripted_reward, ls="-.", color="#8172b3", label=f"Scripted ({scripted_reward:.1f})")
    ax.axhline(
        random_reward, ls=":", color="#55a868", label=f"Random ({random_reward:.1f})"
    )

    ax.set_xlabel("Timesteps T")
    ax.set_ylabel("Mean episode reward")
    ax.set_title("SNN performance vs. number of timesteps")
    ax.set_xscale("log", base=2)
    ax.set_xticks(T_list)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_action_agreement_vs_T(
    T_list: list[int],
    agreements: list[float],
    out_path: str = "results/results_agreement_vs_T.png",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(
        T_list,
        [a * 100 for a in agreements],
        marker="s",
        color="#c44e52",
        label="Action agreement",
    )
    ax.axhline(90, ls="--", color="gray", label="90% target")
    ax.set_xlabel("Timesteps T")
    ax.set_ylabel("Action agreement with ANN (%)")
    ax.set_title("ANN vs. SNN action agreement")
    ax.set_xscale("log", base=2)
    ax.set_xticks(T_list)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(0, 105)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_spike_rates(
    layer_names: list[str],
    firing_rates: list[float],
    out_path: str = "results/results_spike_rates.png",
    T: int | None = None,
    total_spikes: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
    bars = ax.bar(
        layer_names, firing_rates, color=colors[: len(layer_names)], edgecolor="white"
    )
    max_rate = max(firing_rates) if firing_rates else 1.0
    for bar, rate in zip(bars, firing_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(max_rate * 0.02, 0.002),
            f"{rate:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Mean firing rate (spikes / neuron / timestep)")
    title = "Spike activity per layer"
    if T is not None and total_spikes is not None:
        title += f" (T={T}, {total_spikes:,.0f} spikes/decision)"
    elif T is not None:
        title += f" (T={T})"
    ax.set_title(title)
    ax.set_ylim(0, max(max_rate * 1.25, 0.05))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Agreement & Sweep helpers ─────────────────────────────────────────────────

def measure_agreement(
    ann: ANNAgent,
    snn_path: str,
    T: int,
    device: torch.device,
    n_episodes: int = 30,
    seed: int = 0,
) -> float:
    from envs.doom_env import DoomEnv

    snn_agent = SNNAgent(snn_path, device, T=T)
    env = DoomEnv(seed=seed)
    agree = total = 0
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            a_ann = ann.act(obs)
            a_snn = snn_agent.act(obs, T=T)
            agree += int(a_ann == a_snn)
            total += 1
            obs, _, done = env.step(a_ann)
    env.close()
    return agree / max(total, 1)


def select_operating_T(
    T_values: list[int],
    snn_rewards: list[float],
    ann_reward: float,
    threshold_frac: float = 0.95,
) -> int:
    """
    Pick the smallest T achieving >= threshold_frac of ANN reward
    (handles negative rewards safely; falls back to argmax).
    """
    target = ann_reward * threshold_frac if ann_reward >= 0 else ann_reward / threshold_frac
    for T, r in sorted(zip(T_values, snn_rewards), key=lambda x: x[0]):
        if r >= target:
            return T
    return T_values[int(np.argmax(snn_rewards))]


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ann-checkpoint", type=str, default="checkpoints/best.pt")
    p.add_argument("--snn-checkpoint", type=str, default="checkpoints/best_snn.pt")
    p.add_argument("--T-values",       type=int, nargs="+", default=T_VALUES)
    p.add_argument("--episodes",       type=int, default=50)
    p.add_argument("--agreement-eps",  type=int, default=20)
    p.add_argument("--spike-T",        type=int, default=None,
                   help="Override T for spike-rate plot (default: smallest T >= 95%% ANN reward)")
    p.add_argument("--out-dir",        type=str, default="results")
    p.add_argument("--seed",           type=int, default=0)
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print("Evaluating baselines …")
    random_stats   = run_evaluation(RandomAgent(),   n_episodes=args.episodes, seed=args.seed, verbose=False)
    scripted_stats = run_evaluation(ScriptedAgent(), n_episodes=args.episodes, seed=args.seed, verbose=False)
    ann_agent      = ANNAgent(args.ann_checkpoint, device)
    ann_stats      = run_evaluation(ann_agent,       n_episodes=args.episodes, seed=args.seed, verbose=False)

    print(f"  Random   : {random_stats['mean_reward']:.1f}  (kill={random_stats['kill_rate']*100:.1f}%)")
    print(f"  Scripted : {scripted_stats['mean_reward']:.1f}  (kill={scripted_stats['kill_rate']*100:.1f}%)")
    print(f"  ANN      : {ann_stats['mean_reward']:.1f}  (kill={ann_stats['kill_rate']*100:.1f}%)")

    print("\nEvaluating SNN at each T …")
    snn_rewards, snn_stds, agreements, snn_kill_rates = [], [], [], []
    snn_agent = SNNAgent(args.snn_checkpoint, device, T=args.T_values[0])

    for T in args.T_values:
        stats = run_evaluation(snn_agent, n_episodes=args.episodes, seed=args.seed, T=T, verbose=False)
        agreement = measure_agreement(
            ann_agent, args.snn_checkpoint, T=T, device=device,
            n_episodes=args.agreement_eps, seed=args.seed,
        )
        print(
            f"  T={T:4d}  reward={stats['mean_reward']:7.1f} ± {stats['std_reward']:.1f}  "
            f"kill={stats['kill_rate']*100:5.1f}%  agreement={agreement * 100:.1f}%"
        )
        snn_rewards.append(stats["mean_reward"])
        snn_stds.append(stats["std_reward"])
        snn_kill_rates.append(stats["kill_rate"])
        agreements.append(agreement)

    # ── Chart 1: Reward vs T ──────────────────────────────────────────────────
    plot_reward_vs_T(
        ann_reward=ann_stats["mean_reward"],
        random_reward=random_stats["mean_reward"],
        scripted_reward=scripted_stats["mean_reward"],
        T_list=args.T_values,
        snn_rewards=snn_rewards,
        snn_stds=snn_stds,
        out_path=os.path.join(args.out_dir, "results_reward_vs_T.png"),
    )

    # ── Chart 2: Action Agreement vs T ────────────────────────────────────────
    plot_action_agreement_vs_T(
        T_list=args.T_values,
        agreements=agreements,
        out_path=os.path.join(args.out_dir, "results_agreement_vs_T.png"),
    )

    # ── Chart 3: Spike Rates per Layer ────────────────────────────────────────
    selected_T = (
        args.spike_T
        if args.spike_T is not None
        else select_operating_T(args.T_values, snn_rewards, ann_stats["mean_reward"])
    )
    print(f"\nCollecting spike stats at selected T={selected_T} …")
    from analysis.spike_stats import analyse_episodes

    spike_summary = analyse_episodes(
        snn_path=args.snn_checkpoint,
        n_episodes=min(args.episodes, 20),
        T=selected_T,
        device=device,
        seed=args.seed,
    )
    layer_names  = list(spike_summary["layers"].keys())
    firing_rates = [spike_summary["layers"][k]["firing_rate"] for k in layer_names]
    plot_spike_rates(
        layer_names=layer_names,
        firing_rates=firing_rates,
        out_path=os.path.join(args.out_dir, "results_spike_rates.png"),
        T=selected_T,
        total_spikes=spike_summary["mean_spikes_per_decision"],
    )

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        "baselines": {
            "random":   random_stats,
            "scripted": scripted_stats,
            "ann":      ann_stats,
        },
        "t_sweep": [
            {
                "T":            T,
                "mean_reward":  r,
                "std_reward":   s,
                "kill_rate":    kr,
                "agreement":    ag,
            }
            for T, r, s, kr, ag in zip(args.T_values, snn_rewards, snn_stds, snn_kill_rates, agreements)
        ],
        "selected_T":    selected_T,
        "spike_summary": spike_summary,
    }
    summary_path = os.path.join(args.out_dir, "results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")
    print("\nAll plots and summary saved to:", args.out_dir)


if __name__ == "__main__":
    main()
