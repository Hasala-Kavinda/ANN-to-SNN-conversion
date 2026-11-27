"""
Record an MP4 video of one or more episodes.

Uses a single synchronised ViZDoom instance (`DoomEnv(capture_rgb=True)`)
capturing every game tic at native 35 fps, with an optional side-by-side
preprocessed agent view and HUD overlay.

Usage
-----
# Random agent
python evaluation/record_video.py --mode random

# ANN agent (3 episodes, side-by-side + HUD)
python evaluation/record_video.py --mode ann --checkpoint checkpoints/best.pt --episodes 3

# SNN agent
python evaluation/record_video.py --mode snn --checkpoint checkpoints/best_snn.pt --T 32 --episodes 3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import VIDEO_DIR
from envs.doom_env import DoomEnv

try:
    import imageio
    import imageio.plugins.ffmpeg  # noqa: F401
    _IMAGEIO_OK = True
except ImportError:
    _IMAGEIO_OK = False
    print("[Warning] imageio or imageio-ffmpeg not available — saving GIF instead.")

ACTION_NAMES = ["LEFT", "RIGHT", "ATTACK"]


def _make_agent(mode: str, checkpoint: str | None, device: torch.device, T: int):
    from evaluation.eval_agent import ANNAgent, RandomAgent, ScriptedAgent, SNNAgent
    if mode == "random":
        return RandomAgent()
    elif mode == "scripted":
        return ScriptedAgent()
    elif mode == "ann":
        assert checkpoint, "--checkpoint required for ann"
        return ANNAgent(checkpoint, device)
    elif mode == "snn":
        assert checkpoint, "--checkpoint required for snn"
        return SNNAgent(checkpoint, device, T=T)
    raise ValueError(mode)


def _compose_frame(
    rgb:          np.ndarray,
    agent_obs:    np.ndarray,
    side_by_side: bool,
    overlay:      bool,
    mode_label:   str,
    ep:           int,
    step:         int,
    action:       int,
    reward:       float,
    q_vals:       np.ndarray | None,
) -> np.ndarray:
    """Build a single composited RGB frame with optional agent view and HUD."""
    h, w, _ = rgb.shape

    if side_by_side:
        # Latest preprocessed grayscale frame (84x84 in [0,1]) upscaled to (h, h)
        gray_u8 = np.clip(agent_obs[-1] * 255.0, 0, 255).astype(np.uint8)
        agent_vis = cv2.resize(gray_u8, (h, h), interpolation=cv2.INTER_NEAREST)
        agent_rgb = cv2.cvtColor(agent_vis, cv2.COLOR_GRAY2RGB)
        canvas = np.concatenate([rgb, agent_rgb], axis=1)
    else:
        canvas = rgb.copy()

    if overlay:
        bar_h = 44
        bar = np.zeros((bar_h, canvas.shape[1], 3), dtype=np.uint8)
        bar[:] = (24, 24, 30)

        act_name = ACTION_NAMES[action] if 0 <= action < len(ACTION_NAMES) else str(action)
        line1 = f"{mode_label} | Ep {ep} | Step {step:3d} | Act: {act_name:<6s} | R: {reward:6.1f}"
        cv2.putText(
            bar, line1, (8, 17),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 240), 1, cv2.LINE_AA,
        )
        if q_vals is not None:
            q_str = "Q: " + "  ".join(f"{ACTION_NAMES[i]}={q_vals[i]:+.3f}" for i in range(len(q_vals)))
        else:
            q_str = "Q: n/a (heuristic/random)"
        cv2.putText(
            bar, q_str, (8, 36),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 210, 255), 1, cv2.LINE_AA,
        )
        canvas = np.concatenate([canvas, bar], axis=0)

    return canvas


def record_episode(
    agent,
    mode:         str = "random",
    T:            int = 32,
    fps:          int = 35,
    episodes:     int = 3,
    side_by_side: bool = True,
    overlay:      bool = True,
    out_dir:      str = VIDEO_DIR,
    seed:         int = 0,
) -> str:
    """
    Play `episodes` full episodes in a single synchronised DoomEnv and save as MP4 (or GIF).

    Returns the path to the saved file.
    """
    os.makedirs(out_dir, exist_ok=True)
    mode_label = f"SNN (T={T})" if mode == "snn" else mode.upper()

    env = DoomEnv(seed=seed, capture_rgb=True)
    frames_out: list[np.ndarray] = []

    for ep in range(1, episodes + 1):
        obs  = env.reset()
        done = False
        ep_r = 0.0
        step = 0

        while not done:
            q_vals = None
            if hasattr(agent, "act_with_q"):
                action, q_vals = agent.act_with_q(obs, T=T)
            else:
                action = agent.act(obs, T=T)

            prev_obs = obs
            obs, r, done = env.step(action)
            ep_r += r
            step += 1

            for rgb in env.last_step_rgb_frames:
                frames_out.append(
                    _compose_frame(
                        rgb=rgb,
                        agent_obs=prev_obs,
                        side_by_side=side_by_side,
                        overlay=overlay,
                        mode_label=mode_label,
                        ep=ep,
                        step=step,
                        action=action,
                        reward=ep_r,
                        q_vals=q_vals,
                    )
                )

    env.close()

    tag      = f"{mode}_T{T}" if mode == "snn" else mode
    out_path = os.path.join(out_dir, f"{tag}_episode.mp4")

    if _IMAGEIO_OK and frames_out:
        writer = imageio.get_writer(out_path, fps=fps, macro_block_size=1)
        for f in frames_out:
            writer.append_data(f)
        writer.close()
        print(f"Video saved: {out_path}  ({len(frames_out)} frames @ {fps} fps, {episodes} ep)")
    else:
        out_path = out_path.replace(".mp4", ".gif")
        if frames_out:
            imageio.mimsave(out_path, frames_out, fps=fps)
            print(f"GIF saved:   {out_path}  ({len(frames_out)} frames @ {fps} fps, {episodes} ep)")

    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record DOOM agent gameplay video")
    p.add_argument("--mode",          choices=["random", "scripted", "ann", "snn"], default="random")
    p.add_argument("--checkpoint",    type=str, default=None)
    p.add_argument("--T",             type=int, default=32)
    p.add_argument("--fps",           type=int, default=35)
    p.add_argument("--episodes",      type=int, default=3)
    p.add_argument("--out-dir",       type=str, default=VIDEO_DIR)
    p.add_argument("--no-side-by-side", action="store_false", dest="side_by_side", default=True)
    p.add_argument("--no-overlay",      action="store_false", dest="overlay", default=True)
    p.add_argument("--seed",          type=int, default=0)
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent  = _make_agent(args.mode, args.checkpoint, device, args.T)
    record_episode(
        agent,
        mode=args.mode,
        T=args.T,
        fps=args.fps,
        episodes=args.episodes,
        side_by_side=args.side_by_side,
        overlay=args.overlay,
        out_dir=args.out_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
