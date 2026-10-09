"""
Main DQN training entry point for VizDoom basic scenario.

Usage
-----
Full training on server:
    python training/train_doom.py

Resume from checkpoint:
    python training/train_doom.py --resume checkpoints/step_100000.pt

Quick local smoke-test:
    python training/train_doom.py --max-steps 5000

View training curves:
    tensorboard --logdir runs/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import (
    CHECKPOINT_DIR,
    EVAL_EPISODES,
    EVAL_EVERY,
    LOG_DIR,
    MAX_STEPS,
    SAVE_EVERY,
    SEED,
    WARMUP_STEPS,
)
from envs.doom_env import DoomEnv
from training.dqn_agent import DQNAgent

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False
    print("[Warning] TensorBoard not available — logging disabled.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DQN on VizDoom basic")
    p.add_argument("--resume",    type=str,  default=None,
                   help="Checkpoint path to resume from")
    p.add_argument("--max-steps", type=int,  default=MAX_STEPS,
                   help="Override total training steps")
    p.add_argument("--seed",      type=int,  default=SEED)
    p.add_argument("--no-tb",     action="store_true",
                   help="Disable TensorBoard")
    return p.parse_args()


# ── Evaluation helper ─────────────────────────────────────────────────────────

def evaluate_greedy(agent: DQNAgent, n_episodes: int = EVAL_EPISODES) -> dict:
    """Run n episodes with greedy policy; return reward/step statistics."""
    env     = DoomEnv()
    rewards = []
    steps   = []

    for _ in range(n_episodes):
        obs  = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            action        = agent.select_action(obs, greedy=True)
            obs, r, done  = env.step(action)
            ep_r += r
        rewards.append(ep_r)
        steps.append(env.episode_steps)

    env.close()
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward":  float(np.std(rewards)),
        "mean_steps":  float(np.mean(steps)),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU  : {props.name}")
        print(f"  VRAM : {props.total_memory / 1e9:.1f} GB")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,        exist_ok=True)

    # TensorBoard
    writer = None
    if _TB_AVAILABLE and not args.no_tb:
        run_name = f"dqn_{time.strftime('%Y%m%d_%H%M%S')}"
        writer   = SummaryWriter(log_dir=os.path.join(LOG_DIR, run_name))
        print(f"TensorBoard: tensorboard --logdir {LOG_DIR}  (run={run_name})")

    agent = DQNAgent(device)
    env   = DoomEnv(seed=args.seed)

    if args.resume:
        agent.load(args.resume)

    max_steps   = args.max_steps
    best_reward = -float("inf")

    obs  = env.reset()
    done = False

    ep_reward  = 0.0
    ep_count   = 0
    loss_buf   : list[float] = []
    ep_start   = time.time()

    pbar = tqdm(
        total=max_steps,
        initial=agent.steps_done,
        unit="step",
        dynamic_ncols=True,
    )

    while agent.steps_done < max_steps:

        # ── End of episode bookkeeping ────────────────────────────────────────
        if done:
            ep_count += 1
            eps       = agent.epsilon

            pbar.set_postfix(
                ep    = ep_count,
                R     = f"{ep_reward:.0f}",
                best  = f"{best_reward:.0f}",
                eps   = f"{eps:.3f}",
                loss  = f"{np.mean(loss_buf):.4f}" if loss_buf else "n/a",
            )

            if writer:
                writer.add_scalar("train/episode_reward", ep_reward,       agent.steps_done)
                writer.add_scalar("train/epsilon",        eps,             agent.steps_done)
                writer.add_scalar("train/episode_steps",  env.episode_steps, agent.steps_done)
                if loss_buf:
                    writer.add_scalar("train/loss", float(np.mean(loss_buf)), agent.steps_done)

            # Save new best checkpoint
            if ep_reward > best_reward:
                best_reward = ep_reward
                agent.save(os.path.join(CHECKPOINT_DIR, "best.pt"))

            ep_reward = 0.0
            loss_buf  = []
            ep_start  = time.time()
            obs       = env.reset()
            done      = False

        # ── Step ──────────────────────────────────────────────────────────────
        action          = agent.select_action(obs)
        next_obs, r, done = env.step(action)

        agent.store(obs, action, r, next_obs, done)
        ep_reward += r
        obs        = next_obs

        # ── Learn ─────────────────────────────────────────────────────────────
        loss = agent.learn()
        if loss is not None:
            loss_buf.append(loss)

        pbar.update(1)

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if agent.steps_done % SAVE_EVERY == 0 and agent.steps_done > 0:
            ckpt = os.path.join(CHECKPOINT_DIR, f"step_{agent.steps_done}.pt")
            agent.save(ckpt)

        # ── Periodic evaluation ───────────────────────────────────────────────
        if (
            agent.steps_done % EVAL_EVERY == 0
            and agent.steps_done >= WARMUP_STEPS
        ):
            stats = evaluate_greedy(agent)
            print(
                f"\n[Eval @ step {agent.steps_done:,}]  "
                f"mean_reward={stats['mean_reward']:.1f} "
                f"± {stats['std_reward']:.1f}  "
                f"mean_steps={stats['mean_steps']:.0f}"
            )
            if writer:
                for k, v in stats.items():
                    writer.add_scalar(f"eval/{k}", v, agent.steps_done)

    pbar.close()
    env.close()

    # Final save
    final_path = os.path.join(CHECKPOINT_DIR, "final.pt")
    agent.save(final_path)
    print(f"\nTraining complete.")
    print(f"  Total steps  : {agent.steps_done:,}")
    print(f"  Best reward  : {best_reward:.1f}")
    print(f"  Checkpoints  : {CHECKPOINT_DIR}/")

    if writer:
        writer.close()


if __name__ == "__main__":
    main()
