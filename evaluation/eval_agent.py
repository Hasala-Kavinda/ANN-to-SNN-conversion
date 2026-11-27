"""
Evaluation script — reusable across all agent types.

Supports:
  - Random agent (baseline floor)
  - Scripted agent (heuristic ceiling)
  - ANN DQN agent
  - SNN agent (with configurable T)

Usage
-----
# Random baseline
python evaluation/eval_agent.py --mode random --episodes 100

# ANN agent
python evaluation/eval_agent.py --mode ann --checkpoint checkpoints/best.pt

# SNN agent at T=32
python evaluation/eval_agent.py --mode snn --checkpoint checkpoints/best_snn.pt --T 32

# Scripted heuristic agent
python evaluation/eval_agent.py --mode scripted --episodes 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.stats import bootstrap_ci, paired_summary
from configs.config import FRAME_STACK, IMG_H, IMG_W, N_ACTIONS
from envs.doom_env import DoomEnv


# ── Agent wrappers ────────────────────────────────────────────────────────────

class RandomAgent:
    """Uniformly random action selection — baseline floor."""
    def __init__(self):
        self.n_actions = N_ACTIONS

    def act(self, obs: np.ndarray, **_) -> int:
        return int(np.random.randint(self.n_actions))


class ScriptedAgent:
    """
    Simple heuristic agent — rough performance ceiling.

    Reads the center column of the bottom half of the latest frame.
    Turns left/right to centre the brightest region (the monster),
    then attacks once centred. This mimics what an optimal policy does
    but without learning.
    """

    def __init__(self, centre_band: int = 10) -> None:
        self.centre_band = centre_band

    def act(self, obs: np.ndarray, **_) -> int:
        frame  = obs[-1]                         # latest frame, shape (H, W)
        h, w   = frame.shape
        col    = frame[h // 2:, :]               # bottom half
        bright = col.mean(axis=0)                # column brightness profile
        peak   = int(np.argmax(bright))
        centre = w // 2

        if abs(peak - centre) < self.centre_band:
            return 2   # ATTACK — monster centred
        elif peak < centre:
            return 0   # MOVE_LEFT
        else:
            return 1   # MOVE_RIGHT


class ANNAgent:
    """Greedy DQN agent."""

    def __init__(self, checkpoint: str, device: torch.device) -> None:
        from models.ann import DQN
        self.device = device
        ckpt        = torch.load(checkpoint, map_location=device)
        # Support both raw state_dict and full agent checkpoint
        sd = ckpt.get("online_net", ckpt)
        in_ch = int(sd["conv.0.weight"].shape[1]) if "conv.0.weight" in sd else FRAME_STACK
        if in_ch != FRAME_STACK:
            raise ValueError(
                f"ANN checkpoint expects in_channels={in_ch}, "
                f"but configs.config.FRAME_STACK={FRAME_STACK}."
            )
        self.net = DQN(in_channels=in_ch).to(device)
        self.net.load_state_dict(sd)
        self.net.eval()

    def act(self, obs: np.ndarray, **_) -> int:
        action, _ = self.act_with_q(obs)
        return action

    def act_with_q(self, obs: np.ndarray, **_) -> tuple[int, np.ndarray]:
        obs_t = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q = self.net(obs_t).squeeze(0)
            return int(q.argmax().item()), q.cpu().numpy()


class SNNAgent:
    """Converted SNN agent."""

    def __init__(
        self, checkpoint: str, device: torch.device, T: int = 32
    ) -> None:
        from models.snn import SNNFromDQN
        self.device = device
        self.T      = T
        self.net    = SNNFromDQN.load_checkpoint(checkpoint, device=device)

    def act(self, obs: np.ndarray, T: int | None = None, **_) -> int:
        action, _ = self.act_with_q(obs, T=T)
        return action

    def act_with_q(self, obs: np.ndarray, T: int | None = None, **_) -> tuple[int, np.ndarray]:
        t     = T if T is not None else self.T
        obs_t = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q = self.net(obs_t, T=t).squeeze(0)
            return int(q.argmax().item()), q.cpu().numpy()


# ── Core evaluation loop ──────────────────────────────────────────────────────

def run_evaluation(
    agent,
    n_episodes: int = 100,
    seed:       int = 0,
    T:          int = 32,
    verbose:    bool = True,
    paired:     bool = False,
) -> dict:
    """
    Run n_episodes episodes with the given agent.

    If `paired=True`, each episode ep creates a fresh DoomEnv(seed=seed + ep)
    so trajectories across different agents start from identical scenarios.
    """
    rewards, steps_list, kills_list = [], [], []
    env = None if paired else DoomEnv(seed=seed)

    try:
        for ep in range(n_episodes):
            if paired:
                env = DoomEnv(seed=seed + ep)
            obs  = env.reset()
            done = False
            ep_r = 0.0

            while not done:
                action       = agent.act(obs, T=T)
                obs, r, done = env.step(action)
                ep_r        += r

            rewards.append(ep_r)
            steps_list.append(env.episode_steps)
            kills_list.append(env.episode_kills > 0)

            if verbose and (ep + 1) % 10 == 0:
                print(f"  Episode {ep+1:3d}/{n_episodes}  "
                      f"reward={ep_r:7.1f}  steps={env.episode_steps}  "
                      f"kills={env.episode_kills}")

            if paired:
                env.close()
                env = None
    finally:
        if env is not None:
            env.close()

    rewards_arr = np.array(rewards, dtype=np.float64)
    steps_arr   = np.array(steps_list, dtype=np.float64)
    kills_arr   = np.array(kills_list, dtype=np.float64)
    kill_rate   = float(np.mean(kills_arr))
    pos_rate    = float(np.mean(rewards_arr > 0))
    r_ci_lo, r_ci_hi = bootstrap_ci(rewards_arr)
    k_ci_lo, k_ci_hi = bootstrap_ci(kills_arr)

    stats = {
        "n_episodes":           n_episodes,
        "paired":               paired,
        "mean_reward":          float(np.mean(rewards_arr)),
        "std_reward":           float(np.std(rewards_arr)),
        "reward_ci_lo":         r_ci_lo,
        "reward_ci_hi":         r_ci_hi,
        "min_reward":           float(np.min(rewards_arr)),
        "max_reward":           float(np.max(rewards_arr)),
        "mean_steps":           float(np.mean(steps_arr)),
        "std_steps":            float(np.std(steps_arr)),
        "kill_rate":            kill_rate,
        "kill_rate_ci_lo":      k_ci_lo,
        "kill_rate_ci_hi":      k_ci_hi,
        "positive_reward_rate": pos_rate,
        "rewards":              [float(r) for r in rewards_arr],
    }

    if verbose:
        _print_stats(stats)

    return stats


def compare_paired(ann_stats: dict, snn_stats: dict) -> dict:
    """Compute paired (SNN - ANN) difference + 95% bootstrap CI from run_evaluation dicts."""
    return paired_summary(ann_stats["rewards"], snn_stats["rewards"])


def _print_stats(stats: dict) -> None:
    print("\n── Evaluation Results ─────────────────────────────────")
    print(
        f"  Mean reward : {stats['mean_reward']:8.2f} ± {stats['std_reward']:.2f}  "
        f"(95% CI [{stats['reward_ci_lo']:.2f}, {stats['reward_ci_hi']:.2f}])"
    )
    print(f"  Min / Max   : {stats['min_reward']:.1f} / {stats['max_reward']:.1f}")
    print(f"  Mean steps  : {stats['mean_steps']:.1f} ± {stats['std_steps']:.1f}")
    print(
        f"  Kill rate   : {stats['kill_rate']*100:.1f}%  "
        f"(95% CI [{stats['kill_rate_ci_lo']*100:.1f}%, {stats['kill_rate_ci_hi']*100:.1f}%], "
        f"reward > 0: {stats['positive_reward_rate']*100:.1f}%)"
    )
    print("────────────────────────────────────────────────────────")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a DOOM agent")
    p.add_argument("--mode",       choices=["random", "scripted", "ann", "snn"],
                   default="random")
    p.add_argument("--checkpoint", type=str,  default=None,
                   help="Path to .pt checkpoint (required for ann/snn)")
    p.add_argument("--config",     type=str,  default=None,
                   help="Optional chosen_config.json (supplies T when --T is not set)")
    p.add_argument("--episodes",   type=int,  default=100)
    p.add_argument("--T",          type=int,  default=None,
                   help="SNN timesteps (default: from --config or 32)")
    p.add_argument("--seed",       type=int,  default=0)
    p.add_argument("--paired",     action="store_true",
                   help="Use identical per-episode seeds (seed+ep) for paired comparison")
    p.add_argument("--json-out",   type=str,  default=None,
                   help="Optional path to save evaluation JSON")
    p.add_argument("--quiet",      action="store_true")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    T = args.T
    if T is None and args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        T = int(cfg.get("T", 32))
    if T is None:
        T = 32

    print(f"Mode: {args.mode}  |  Episodes: {args.episodes}  |  Paired: {args.paired}  |  Device: {device}")

    if args.mode == "random":
        agent = RandomAgent()
    elif args.mode == "scripted":
        agent = ScriptedAgent()
    elif args.mode == "ann":
        assert args.checkpoint, "--checkpoint required for --mode ann"
        agent = ANNAgent(args.checkpoint, device)
    elif args.mode == "snn":
        assert args.checkpoint, "--checkpoint required for --mode snn"
        agent = SNNAgent(args.checkpoint, device, T=T)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    stats = run_evaluation(
        agent,
        n_episodes=args.episodes,
        seed=args.seed,
        T=T,
        verbose=not args.quiet,
        paired=args.paired,
    )
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved evaluation JSON: {args.json_out}")


if __name__ == "__main__":
    main()
