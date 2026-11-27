"""
Two-stage ANN-to-SNN conversion parameter sweep.

Stage 1 (Offline Screen, seconds-to-minutes):
  Evaluates every (percentile, mem_init, T) combination in batched mode on
  saved calibration frames (calibration_frames.npz), measuring action agreement,
  decision regret, Q-MAE vs ANN Q-gap, clip fraction, and dead neurons.

Stage 2 (Gameplay Confirmation):
  Plays paired-seed ViZDoom episodes for only the top-K Stage 1 candidates and
  selects the smallest T achieving >= 95% of ANN reward and >= 95% agreement.

Outputs (in --out-dir, default results/):
  - sweep_results.csv
  - sweep_heatmap.png
  - chosen_config.json

Usage
-----
python conversion/sweep.py \
    --checkpoint checkpoints/best.pt \
    --stats calibration_stats.pkl \
    --frames calibration_frames.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import FRAME_STACK, T_VALUES
from analysis.stats import bootstrap_ci as _bootstrap_ci
from conversion.convert import build_snn_from_ann
from conversion.normalize import compute_thresholds
from envs.doom_env import DoomEnv
from models.ann import DQN


def screen_offline(
    ann:         DQN,
    stats:       dict[str, np.ndarray],
    frames_u8:   np.ndarray,
    percentiles: list[float],
    mem_inits:   list[float],
    T_values:    list[int],
    device:      torch.device,
    batch_size:  int = 128,
) -> list[dict]:
    """
    Stage 1: Batched offline evaluation of (percentile, mem_init, T) on saved frames.
    """
    ann.to(device).eval()
    frames_f32 = torch.from_numpy(frames_u8).float().div_(255.0)
    N = frames_f32.shape[0]

    # Precompute ANN Q-values once
    q_ann_chunks = []
    with torch.no_grad():
        for start in range(0, N, batch_size):
            xb = frames_f32[start : start + batch_size].to(device)
            q_ann_chunks.append(ann(xb).cpu())
    q_ann = torch.cat(q_ann_chunks, dim=0)
    a_ann = q_ann.argmax(dim=1)
    q_max = q_ann.max(dim=1).values
    sorted_q, _ = torch.sort(q_ann, dim=1, descending=True)
    ann_q_gap = float((sorted_q[:, 0] - sorted_q[:, 1]).mean().item())

    rows: list[dict] = []
    sorted_layers = sorted(stats.keys())

    for pct in percentiles:
        thresh_dict, clip_dict = compute_thresholds(
            stats, pct, verbose=False, return_clip_fracs=True
        )
        thresholds = [thresh_dict[k] for k in sorted_layers]
        mean_clip = float(np.mean(list(clip_dict.values())))
        max_clip  = float(np.max(list(clip_dict.values())))

        for mem_init in mem_inits:
            snn = build_snn_from_ann(ann, thresholds, mem_init_ratio=mem_init).to(device).eval()

            for T in T_values:
                q_snn_chunks = []
                dead_fracs = []
                with torch.no_grad():
                    for b_idx, start in enumerate(range(0, N, batch_size)):
                        xb = frames_f32[start : start + batch_size].to(device)
                        if b_idx == 0:
                            q_b, sp_stats = snn.forward_with_stats(xb, T=T)
                            dead_fracs = [
                                sp_stats[k]["silent_neuron_frac"]
                                for k in sp_stats if not k.startswith("_")
                            ]
                        else:
                            q_b = snn(xb, T=T)
                        q_snn_chunks.append(q_b.cpu())

                q_snn = torch.cat(q_snn_chunks, dim=0)
                a_snn = q_snn.argmax(dim=1)

                agreement = float((a_snn == a_ann).float().mean().item())
                regret    = float((q_max - q_ann.gather(1, a_snn.unsqueeze(1)).squeeze(1)).mean().item())
                q_mae     = float((q_ann - q_snn).abs().mean().item())
                q_ratio   = q_mae / max(ann_q_gap, 1e-9)

                rows.append({
                    "percentile":       pct,
                    "mem_init":         mem_init,
                    "T":                T,
                    "agreement":        agreement,
                    "regret":           regret,
                    "q_mae":            q_mae,
                    "ann_q_gap":        ann_q_gap,
                    "q_mae_over_gap":   q_ratio,
                    "mean_clip_frac":   mean_clip,
                    "max_clip_frac":    max_clip,
                    "dead_neuron_frac": float(np.mean(dead_fracs)) if dead_fracs else 0.0,
                    "thresholds":       thresholds,
                })
                print(
                    f"  [S1] pct={pct:6.2f}  mem={mem_init:.1f}  T={T:3d}  "
                    f"agree={agreement*100:5.1f}%  regret={regret:.5f}  "
                    f"MAE/gap={q_ratio:.2f}  clip={mean_clip*100:.3f}%"
                )

    return rows


def _eval_paired_episodes(
    agent_fn,
    in_channels: int,
    n_episodes:  int,
) -> tuple[np.ndarray, np.ndarray]:
    rewards = np.zeros(n_episodes, dtype=np.float64)
    kills   = np.zeros(n_episodes, dtype=np.float64)
    for ep in range(n_episodes):
        env = DoomEnv(frame_stack=in_channels, seed=ep)
        obs = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            act = agent_fn(obs)
            obs, r, done = env.step(act)
            ep_r += r
        rewards[ep] = ep_r
        kills[ep]   = float(env.episode_kills > 0)
        env.close()
    return rewards, kills


def confirm_gameplay(
    ann:         DQN,
    in_channels: int,
    candidates:  list[dict],
    n_episodes:  int,
    device:      torch.device,
) -> tuple[dict, list[dict]]:
    """
    Stage 2: Paired-seed gameplay evaluation of top-K Stage 1 candidates.
    """
    ann.to(device).eval()

    def ann_act(obs: np.ndarray) -> int:
        t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            return int(ann(t).argmax(dim=1).item())

    ann_rewards, ann_kills = _eval_paired_episodes(ann_act, in_channels, n_episodes)
    ann_mean = float(ann_rewards.mean())
    ann_ci   = _bootstrap_ci(ann_rewards)
    print(f"\n[S2] ANN baseline ({n_episodes} paired eps): "
          f"reward={ann_mean:.2f}  95% CI=[{ann_ci[0]:.1f}, {ann_ci[1]:.1f}]  "
          f"kill={ann_kills.mean()*100:.1f}%")

    confirmed: list[dict] = []
    for c in candidates:
        snn = build_snn_from_ann(ann, c["thresholds"], mem_init_ratio=c["mem_init"]).to(device).eval()
        T = c["T"]

        def snn_act(obs: np.ndarray, _snn=snn, _T=T) -> int:
            t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                return int(_snn(t, T=_T).argmax(dim=1).item())

        snn_rewards, snn_kills = _eval_paired_episodes(snn_act, in_channels, n_episodes)
        snn_mean = float(snn_rewards.mean())
        snn_ci   = _bootstrap_ci(snn_rewards)
        diff     = snn_rewards - ann_rewards
        diff_ci  = _bootstrap_ci(diff)
        ratio    = (snn_mean / ann_mean) if abs(ann_mean) > 1e-6 else 1.0

        rec = {
            **c,
            "s2_ann_mean":   ann_mean,
            "s2_snn_mean":   snn_mean,
            "s2_snn_ci_lo":  snn_ci[0],
            "s2_snn_ci_hi":  snn_ci[1],
            "s2_paired_diff": float(diff.mean()),
            "s2_diff_ci_lo": diff_ci[0],
            "s2_diff_ci_hi": diff_ci[1],
            "s2_kill_rate":  float(snn_kills.mean()),
            "s2_ratio_ann":  ratio,
        }
        confirmed.append(rec)
        print(
            f"  [S2] pct={c['percentile']:6.2f}  mem={c['mem_init']:.1f}  T={T:3d}  "
            f"reward={snn_mean:6.1f} [{snn_ci[0]:.1f},{snn_ci[1]:.1f}]  "
            f"diff={diff.mean():+5.1f}  kill={snn_kills.mean()*100:5.1f}%"
        )

    # Pick smallest T meeting >= 95% ANN reward (when ANN > 0) and >= 95% agreement; fallback to highest reward
    target_r = ann_mean * 0.95 if ann_mean >= 0 else ann_mean / 0.95
    passing = [
        r for r in confirmed
        if r["s2_snn_mean"] >= target_r and r["agreement"] >= 0.95
    ]
    if passing:
        chosen = min(passing, key=lambda r: (r["T"], r["regret"]))
    else:
        chosen = max(confirmed, key=lambda r: (r["s2_snn_mean"], -r["T"]))

    return chosen, confirmed


def plot_sweep_heatmap(rows: list[dict], out_path: str) -> None:
    best_mem = max(
        set(r["mem_init"] for r in rows),
        key=lambda m: np.mean([r["agreement"] for r in rows if r["mem_init"] == m]),
    )
    sub = [r for r in rows if r["mem_init"] == best_mem]
    pcts = sorted(set(r["percentile"] for r in sub))
    Ts   = sorted(set(r["T"] for r in sub))

    agree_mat  = np.zeros((len(pcts), len(Ts)))
    regret_mat = np.zeros((len(pcts), len(Ts)))
    for r in sub:
        i = pcts.index(r["percentile"])
        j = Ts.index(r["T"])
        agree_mat[i, j]  = r["agreement"] * 100.0
        regret_mat[i, j] = r["regret"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, mat, title, fmt, cmap in [
        (axes[0], agree_mat,  f"Action Agreement % (mem_init={best_mem})", "{:.1f}%", "YlGn"),
        (axes[1], regret_mat, f"Mean Q-Regret (mem_init={best_mem})",      "{:.4f}",  "YlOrRd"),
    ]:
        im = ax.imshow(mat, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(Ts)), [str(t) for t in Ts])
        ax.set_yticks(range(len(pcts)), [str(p) for p in pcts])
        ax.set_xlabel("Timesteps T")
        ax.set_ylabel("Percentile")
        ax.set_title(title)
        for i in range(len(pcts)):
            for j in range(len(Ts)):
                ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center", fontsize=8, color="black")
        fig.colorbar(im, ax=ax, shrink=0.85)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-stage ANN-to-SNN conversion sweep")
    p.add_argument("--checkpoint",  type=str, default="checkpoints/best.pt")
    p.add_argument("--stats",       type=str, default="calibration_stats.pkl")
    p.add_argument("--frames",      type=str, default="calibration_frames.npz")
    p.add_argument("--percentiles", type=float, nargs="+", default=[99.0, 99.5, 99.9, 99.99, 100.0])
    p.add_argument("--mem-inits",   type=float, nargs="+", default=[0.0, 0.5])
    p.add_argument("--T-values",    type=int,   nargs="+", default=T_VALUES)
    p.add_argument("--top-k",       type=int, default=4)
    p.add_argument("--episodes",    type=int, default=50)
    p.add_argument("--batch-size",  type=int, default=128)
    p.add_argument("--out-dir",     type=str, default="results")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    sd   = ckpt.get("online_net", ckpt)
    in_ch = int(sd["conv.0.weight"].shape[1]) if "conv.0.weight" in sd else FRAME_STACK
    ann = DQN(in_channels=in_ch)
    ann.load_state_dict(sd)
    ann.eval()

    with open(args.stats, "rb") as f:
        stats = pickle.load(f)
    frames_u8 = np.load(args.frames)["frames"]

    t0 = time.time()
    print(f"Stage 1: Screening {len(args.percentiles)*len(args.mem_inits)*len(args.T_values)} "
          f"configs on {len(frames_u8)} frames (device={device}) …")
    s1_rows = screen_offline(
        ann, stats, frames_u8,
        args.percentiles, args.mem_inits, args.T_values,
        device=device, batch_size=args.batch_size,
    )
    print(f"Stage 1 finished in {time.time() - t0:.1f}s")

    # Select top-K candidates: prefer passing agreement (>=0.90) with smallest T and lowest regret
    ranked = sorted(
        s1_rows,
        key=lambda r: (0 if r["agreement"] >= 0.90 else 1, r["T"], r["regret"], -r["agreement"]),
    )
    candidates = ranked[: max(1, min(args.top_k, len(ranked)))]

    chosen, confirmed = confirm_gameplay(
        ann, in_ch, candidates, n_episodes=args.episodes, device=device
    )

    # Save CSV
    csv_path = os.path.join(args.out_dir, "sweep_results.csv")
    csv_keys = [k for k in s1_rows[0].keys() if k != "thresholds"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_keys)
        w.writeheader()
        for r in s1_rows:
            w.writerow({k: r[k] for k in csv_keys})
    print(f"Saved: {csv_path}")

    # Save heatmap
    plot_sweep_heatmap(s1_rows, os.path.join(args.out_dir, "sweep_heatmap.png"))

    # Save chosen_config.json
    chosen_out = {
        "percentile":   chosen["percentile"],
        "mem_init":     chosen["mem_init"],
        "T":            chosen["T"],
        "thresholds":   chosen["thresholds"],
        "s1_agreement": chosen["agreement"],
        "s1_regret":    chosen["regret"],
        "s2_snn_mean":  chosen["s2_snn_mean"],
        "s2_ann_mean":  chosen["s2_ann_mean"],
        "s2_kill_rate": chosen["s2_kill_rate"],
        "convert_cmd": (
            f"python conversion/convert.py --checkpoint {args.checkpoint} "
            f"--stats {args.stats} --no-calibrate "
            f"--percentile {chosen['percentile']} --mem-init {chosen['mem_init']} "
            f"--verify-T {chosen['T']}"
        ),
    }
    chosen_path = os.path.join(args.out_dir, "chosen_config.json")
    with open(chosen_path, "w") as f:
        json.dump(chosen_out, f, indent=2)
    print(f"Saved: {chosen_path}")
    print(f"\nRecommended command:\n  {chosen_out['convert_cmd']}")


if __name__ == "__main__":
    main()
