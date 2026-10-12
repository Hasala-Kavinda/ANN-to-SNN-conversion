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
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
        self.net    = DQN().to(device)
        ckpt        = torch.load(checkpoint, map_location=device)
        # Support both raw state_dict and full agent checkpoint
        sd = ckpt.get("online_net", ckpt)
        self.net.load_state_dict(sd)
        self.net.eval()

    def act(self, obs: np.ndarray, **_) -> int:
        obs_t = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            return int(self.net(obs_t).argmax(dim=1).item())


class SNNAgent:
    """Converted SNN agent."""

    def __init__(
        self, checkpoint: str, device: torch.device, T: int = 32
    ) -> None:
        from models.snn import SNNFromDQN
        self.device = device
        self.T      = T
        self.net    = torch.load(checkpoint, map_location=device)
        self.net.eval()

    def act(self, obs: np.ndarray, T: int | None = None, **_) -> int:
        t     = T if T is not None else self.T
        obs_t = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            return int(self.net(obs_t, T=t).argmax(dim=1).item())


# ── Core evaluation loop ──────────────────────────────────────────────────────

def run_evaluation(
    agent,
    n_episodes: int = 100,
    seed:       int = 0,
    T:          int = 32,
    verbose:    bool = True,
) -> dict:
    """
    Run n_episodes episodes with the given agent.

    Returns
    -------
    dict with keys:
      mean_reward, std_reward, min_reward, max_reward,
      mean_steps,  std_steps,
      kill_rate    (fraction of episodes where reward > 0)
    """
    env = DoomEnv(seed=seed)
    rewards, steps_list = [], []

    for ep in range(n_episodes):
        obs  = env.reset()
        done = False
        ep_r = 0.0

        while not done:
            action       = agent.act(obs, T=T)
            obs, r, done = env.step(action)
            ep_r        += r

        rewards.append(ep_r)
        steps_list.append(env.episode_steps)

        if verbose and (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1:3d}/{n_episodes}  "
                  f"reward={ep_r:7.1f}  steps={env.episode_steps}")

    env.close()

    rewards    = np.array(rewards)
    steps_arr  = np.array(steps_list)
    kill_rate  = float(np.mean(rewards > 0))

    stats = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward":  float(np.std(rewards)),
        "min_reward":  float(np.min(rewards)),
        "max_reward":  float(np.max(rewards)),
        "mean_steps":  float(np.mean(steps_arr)),
        "std_steps":   float(np.std(steps_arr)),
        "kill_rate":   kill_rate,
    }

    if verbose:
        _print_stats(stats)

    return stats


def _print_stats(stats: dict) -> None:
    print("\n── Evaluation Results ─────────────────────────────────")
    print(f"  Mean reward : {stats['mean_reward']:8.2f} ± {stats['std_reward']:.2f}")
    print(f"  Min / Max   : {stats['min_reward']:.1f} / {stats['max_reward']:.1f}")
    print(f"  Mean steps  : {stats['mean_steps']:.1f} ± {stats['std_steps']:.1f}")
    print(f"  Kill rate   : {stats['kill_rate']*100:.1f}%")
    print("────────────────────────────────────────────────────────")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a DOOM agent")
    p.add_argument("--mode",       choices=["random", "scripted", "ann", "snn"],
                   default="random")
    p.add_argument("--checkpoint", type=str,  default=None,
                   help="Path to .pt checkpoint (required for ann/snn)")
    p.add_argument("--episodes",   type=int,  default=100)
    p.add_argument("--T",          type=int,  default=32,
                   help="SNN timesteps (only used for --mode snn)")
    p.add_argument("--seed",       type=int,  default=0)
    p.add_argument("--quiet",      action="store_true")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Mode: {args.mode}  |  Episodes: {args.episodes}  |  Device: {device}")

    if args.mode == "random":
        agent = RandomAgent()
    elif args.mode == "scripted":
        agent = ScriptedAgent()
    elif args.mode == "ann":
        assert args.checkpoint, "--checkpoint required for --mode ann"
        agent = ANNAgent(args.checkpoint, device)
    elif args.mode == "snn":
        assert args.checkpoint, "--checkpoint required for --mode snn"
        agent = SNNAgent(args.checkpoint, device, T=args.T)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    run_evaluation(
        agent,
        n_episodes=args.episodes,
        seed=args.seed,
        T=args.T,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
