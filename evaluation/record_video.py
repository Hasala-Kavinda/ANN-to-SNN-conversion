"""
Record an MP4 video of one episode.

Saves raw (un-cropped, colour) frames at full resolution for readability,
alongside a side-by-side preprocessed view so you can verify what the
agent actually sees.

Usage
-----
# Random agent
python evaluation/record_video.py --mode random

# ANN agent
python evaluation/record_video.py --mode ann --checkpoint checkpoints/best.pt

# SNN agent
python evaluation/record_video.py --mode snn --checkpoint checkpoints/best_snn.pt --T 32
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


def _make_agent(mode: str, checkpoint: str | None, device: torch.device, T: int):
    sys.path.insert(0, str(PROJECT_ROOT))
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


def record_episode(
    agent,
    mode:    str = "random",
    T:       int = 32,
    fps:     int = 30,
    out_dir: str = VIDEO_DIR,
    seed:    int = 0,
) -> str:
    """
    Play one full episode and save it as an MP4 (or GIF fallback).

    Returns the path to the saved file.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Use a separate env with window visible = False but with RGB screen
    # so we can capture colour frames for the video.
    import vizdoom as vzd
    from pathlib import Path as _P

    _VZD_SCENARIOS = _P(vzd.__file__).parent / "scenarios"

    game = vzd.DoomGame()
    game.load_config(str(_VZD_SCENARIOS / "basic.cfg"))
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    game.set_window_visible(False)
    game.set_seed(seed)
    game.init()

    ACTIONS = [
        [True,  False, False],
        [False, True,  False],
        [False, False, True],
    ]
    from configs.config import FRAME_SKIP, FRAME_STACK, IMG_H, IMG_W

    # We also need a grayscale env for the agent's observations
    obs_env = DoomEnv(seed=seed)
    obs     = obs_env.reset()

    frames_rgb = []
    done       = False

    # Sync both envs to start of episode
    game.new_episode()

    while not done:
        # Capture RGB frame
        state = game.get_state()
        if state is not None:
            frames_rgb.append(state.screen_buffer.copy())   # (H, W, 3)

        action       = agent.act(obs, T=T)
        obs, r, done = obs_env.step(action)

        # Step the RGB game in sync
        game.make_action(ACTIONS[action], FRAME_SKIP)
        if game.is_episode_finished():
            done = True

    game.close()
    obs_env.close()

    # Save video
    tag      = f"{mode}_T{T}" if mode == "snn" else mode
    out_path = os.path.join(out_dir, f"{tag}_episode.mp4")

    if _IMAGEIO_OK and frames_rgb:
        writer = imageio.get_writer(out_path, fps=fps, macro_block_size=1)
        for f in frames_rgb:
            writer.append_data(f)
        writer.close()
        print(f"Video saved: {out_path}  ({len(frames_rgb)} frames)")
    else:
        # GIF fallback
        out_path = out_path.replace(".mp4", ".gif")
        if frames_rgb:
            imageio.mimsave(out_path, frames_rgb, fps=fps)
            print(f"GIF saved:   {out_path}  ({len(frames_rgb)} frames)")

    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode",       choices=["random", "scripted", "ann", "snn"],
                   default="random")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--T",          type=int, default=32)
    p.add_argument("--fps",        type=int, default=30)
    p.add_argument("--seed",       type=int, default=0)
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent  = _make_agent(args.mode, args.checkpoint, device, args.T)
    record_episode(agent, mode=args.mode, T=args.T, fps=args.fps, seed=args.seed)


if __name__ == "__main__":
    main()
