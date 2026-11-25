"""
Generate all result plots for the report.

Plots produced
--------------
1. T-sweep: SNN reward vs number of timesteps T, with ANN and random baselines.
2. Action agreement vs T.
3. Spike firing rate per layer (bar chart at a selected T).

Usage
-----
# Run after evaluate and spike_stats
python analysis/plot_results.py \
    --ann-checkpoint  checkpoints/best.pt \
    --snn-checkpoint  checkpoints/best_snn.pt \
    --T-values 8 16 32 64 128 \
    --episodes 100
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
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
    out_path: str = "results_reward_vs_T.png",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.errorbar(
        T_list,
        snn_rewards,
        yerr=snn_stds,
        marker="o",
        label="SNN",
        color="#e07b39",
        capsize=4,
    )
    ax.axhline(ann_reward, ls="--", color="#4c72b0", label=f"ANN ({ann_reward:.0f})")
    ax.axhline(
        random_reward, ls=":", color="#55a868", label=f"Random ({random_reward:.0f})"
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
    out_path: str = "results_agreement_vs_T.png",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
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
    out_path: str = "results_spike_rates.png",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
    bars = ax.bar(
        layer_names, firing_rates, color=colors[: len(layer_names)], edgecolor="white"
    )
    for bar, rate in zip(bars, firing_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{rate:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Mean firing rate (spikes / neuron / timestep)")
    ax.set_title("Spike activity per layer")
    ax.set_ylim(0, max(firing_rates) * 1.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


# Agreement helper


def measure_agreement(
    ann: ANNAgent,
    snn_path: str,
    T: int,
    device: torch.device,
    n_episodes: int = 30,
) -> float:
    from envs.doom_env import DoomEnv

    snn_agent = SNNAgent(snn_path, device, T=T)
    env = DoomEnv()
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


# ── Main ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ann-checkpoint", type=str, default="checkpoints/best.pt")
    p.add_argument("--snn-checkpoint", type=str, default="checkpoints/best_snn.pt")
    p.add_argument("--T-values", type=int, nargs="+", default=T_VALUES)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--out-dir", type=str, default=".")
    return p.parse_args()


def main() -> None:
    import matplotlib.ticker

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print("Evaluating baselines …")
    random_stats = run_evaluation(
        RandomAgent(), n_episodes=args.episodes, verbose=False
    )
    ann_agent = ANNAgent(args.ann_checkpoint, device)
    ann_stats = run_evaluation(ann_agent, n_episodes=args.episodes, verbose=False)

    print(f"  Random  : {random_stats['mean_reward']:.1f}")
    print(f"  ANN     : {ann_stats['mean_reward']:.1f}")

    print("\nEvaluating SNN at each T …")
    snn_rewards, snn_stds, agreements = [], [], []

    for T in args.T_values:
        snn_agent = SNNAgent(args.snn_checkpoint, device, T=T)
        stats = run_evaluation(snn_agent, n_episodes=args.episodes, T=T, verbose=False)
        agreement = measure_agreement(
            ann_agent, args.snn_checkpoint, T=T, device=device, n_episodes=20
        )
        print(
            f"  T={T:4d}  reward={stats['mean_reward']:7.1f} ± {stats['std_reward']:.1f}  "
            f"agreement={agreement * 100:.1f}%"
        )
        snn_rewards.append(stats["mean_reward"])
        snn_stds.append(stats["std_reward"])
        agreements.append(agreement)

    # ── Reward plot ───────────────────────────────────────────────────────────
    plot_reward_vs_T(
        ann_reward=ann_stats["mean_reward"],
        random_reward=random_stats["mean_reward"],
        T_list=args.T_values,
        snn_rewards=snn_rewards,
        snn_stds=snn_stds,
        out_path=os.path.join(args.out_dir, "results_reward_vs_T.png"),
    )

    # ── Agreement plot ────────────────────────────────────────────────────────
    plot_action_agreement_vs_T(
        T_list=args.T_values,
        agreements=agreements,
        out_path=os.path.join(args.out_dir, "results_agreement_vs_T.png"),
    )

    # ── Spike rate plot (using spike_stats module at best T) ──────────────────
    best_T = args.T_values[int(np.argmax(snn_rewards))]
    print(f"\nCollecting spike stats at best T={best_T} …")
    from analysis.spike_stats import analyse_episodes

    analyse_episodes(
        snn_path=args.snn_checkpoint,
        n_episodes=20,
        T=best_T,
    )

    print("\nAll plots saved to:", args.out_dir)


if __name__ == "__main__":
    main()
