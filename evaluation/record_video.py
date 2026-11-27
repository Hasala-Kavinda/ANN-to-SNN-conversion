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


def _compose_compare_frame(
    rgb_ann:    np.ndarray,
    rgb_snn:    np.ndarray,
    overlay:    bool,
    T:          int,
    ep:         int,
    step:       int,
    a_ann:      int,
    a_snn:      int,
    r_ann:      float,
    r_snn:      float,
    q_ann:      np.ndarray,
    q_snn:      np.ndarray,
    spikes_snn: float,
    diverged:   bool,
) -> np.ndarray:
    """Build side-by-side ANN vs SNN frame with synchronized HUD."""
    canvas = np.concatenate([rgb_ann, rgb_snn], axis=1)
    if not overlay:
        return canvas

    bar_h = 56
    bar = np.zeros((bar_h, canvas.shape[1], 3), dtype=np.uint8)
    bar[:] = (24, 24, 30)

    act_a = ACTION_NAMES[a_ann] if 0 <= a_ann < len(ACTION_NAMES) else str(a_ann)
    act_s = ACTION_NAMES[a_snn] if 0 <= a_snn < len(ACTION_NAMES) else str(a_snn)
    agree_str = "AGREE" if a_ann == a_snn else "DIFF"
    if diverged:
        agree_str += " (diverged)"
    agree_col = (120, 230, 140) if a_ann == a_snn else (255, 130, 110)

    line1 = (
        f"Ep {ep} Step {step:3d} | ANN: {act_a:<6s} R={r_ann:6.1f} | "
        f"SNN(T={T}): {act_s:<6s} R={r_snn:6.1f} | {agree_str}"
    )
    cv2.putText(bar, line1, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, agree_col, 1, cv2.LINE_AA)

    qa_str = " ".join(f"{ACTION_NAMES[i][0]}:{q_ann[i]:+.2f}" for i in range(len(q_ann)))
    qs_str = " ".join(f"{ACTION_NAMES[i][0]}:{q_snn[i]:+.2f}" for i in range(len(q_snn)))
    line2 = f"ANN Q[{qa_str}]  |  SNN Q[{qs_str}]  |  Spikes: {spikes_snn:,.0f}/dec"
    cv2.putText(bar, line2, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 215, 255), 1, cv2.LINE_AA)

    return np.concatenate([canvas, bar], axis=0)


def record_compare(
    ann,
    snn,
    T:        int = 32,
    fps:      int = 35,
    episodes: int = 3,
    overlay:  bool = True,
    out_dir:  str = VIDEO_DIR,
    seed:     int = 0,
) -> str:
    """
    Play `episodes` paired-seed episodes side-by-side (ANN left, SNN right)
    and save as MP4 (or GIF).
    """
    os.makedirs(out_dir, exist_ok=True)
    frames_out: list[np.ndarray] = []

    for ep in range(1, episodes + 1):
        ep_seed = seed + ep - 1
        env_ann = DoomEnv(seed=ep_seed, capture_rgb=True)
        env_snn = DoomEnv(seed=ep_seed, capture_rgb=True)

        obs_ann = env_ann.reset()
        obs_snn = env_snn.reset()
        done_ann = False
        done_snn = False
        r_ann = 0.0
        r_snn = 0.0
        step = 0
        diverged = False

        last_rgb_ann = np.zeros((240, 320, 3), dtype=np.uint8)
        last_rgb_snn = np.zeros((240, 320, 3), dtype=np.uint8)
        a_ann, q_ann = 0, np.zeros(3, dtype=np.float32)
        a_snn, q_snn = 0, np.zeros(3, dtype=np.float32)
        spikes_snn = 0.0

        while not (done_ann and done_snn):
            step += 1
            if not done_ann:
                a_ann, q_ann = ann.act_with_q(obs_ann)
                obs_ann, ra, done_ann = env_ann.step(a_ann)
                r_ann += ra
                rgb_list_ann = env_ann.last_step_rgb_frames
            else:
                rgb_list_ann = [last_rgb_ann]

            if not done_snn:
                obs_t = torch.as_tensor(obs_snn, dtype=torch.float32, device=snn.device).unsqueeze(0)
                with torch.no_grad():
                    q_snn_t, sp_stats = snn.net.forward_with_stats(obs_t, T=T)
                q_snn = q_snn_t.squeeze(0).cpu().numpy()
                a_snn = int(np.argmax(q_snn))
                spikes_snn = float(sp_stats["_total_spikes_per_decision"])
                obs_snn, rs, done_snn = env_snn.step(a_snn)
                r_snn += rs
                rgb_list_snn = env_snn.last_step_rgb_frames
            else:
                rgb_list_snn = [last_rgb_snn]

            if a_ann != a_snn:
                diverged = True

            n_sub = max(len(rgb_list_ann), len(rgb_list_snn))
            for i in range(n_sub):
                fa = rgb_list_ann[min(i, len(rgb_list_ann) - 1)]
                fs = rgb_list_snn[min(i, len(rgb_list_snn) - 1)]
                last_rgb_ann = fa
                last_rgb_snn = fs
                frames_out.append(
                    _compose_compare_frame(
                        rgb_ann=fa,
                        rgb_snn=fs,
                        overlay=overlay,
                        T=T,
                        ep=ep,
                        step=step,
                        a_ann=a_ann,
                        a_snn=a_snn,
                        r_ann=r_ann,
                        r_snn=r_snn,
                        q_ann=q_ann,
                        q_snn=q_snn,
                        spikes_snn=spikes_snn,
                        diverged=diverged,
                    )
                )

        env_ann.close()
        env_snn.close()

    out_path = os.path.join(out_dir, f"compare_T{T}.mp4")
    if _IMAGEIO_OK and frames_out:
        writer = imageio.get_writer(out_path, fps=fps, macro_block_size=1)
        for f in frames_out:
            writer.append_data(f)
        writer.close()
        print(f"Compare video saved: {out_path}  ({len(frames_out)} frames @ {fps} fps, {episodes} ep)")
    else:
        out_path = out_path.replace(".mp4", ".gif")
        if frames_out:
            imageio.mimsave(out_path, frames_out, fps=fps)
            print(f"Compare GIF saved:   {out_path}  ({len(frames_out)} frames @ {fps} fps, {episodes} ep)")
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record DOOM agent gameplay video")
    p.add_argument("--mode",           choices=["random", "scripted", "ann", "snn"], default="random")
    p.add_argument("--compare",        action="store_true",
                   help="Record side-by-side ANN vs SNN compare video")
    p.add_argument("--checkpoint",     type=str, default=None)
    p.add_argument("--snn-checkpoint", type=str, default=None)
    p.add_argument("--T",              type=int, default=32)
    p.add_argument("--fps",            type=int, default=35)
    p.add_argument("--episodes",       type=int, default=3)
    p.add_argument("--out-dir",        type=str, default=VIDEO_DIR)
    p.add_argument("--no-side-by-side", action="store_false", dest="side_by_side", default=True)
    p.add_argument("--no-overlay",      action="store_false", dest="overlay", default=True)
    p.add_argument("--seed",           type=int, default=0)
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.compare:
        assert args.checkpoint, "--checkpoint (ANN) required for --compare"
        assert args.snn_checkpoint, "--snn-checkpoint required for --compare"
        ann = _make_agent("ann", args.checkpoint, device, args.T)
        snn = _make_agent("snn", args.snn_checkpoint, device, args.T)
        record_compare(
            ann, snn,
            T=args.T,
            fps=args.fps,
            episodes=args.episodes,
            overlay=args.overlay,
            out_dir=args.out_dir,
            seed=args.seed,
        )
        return

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
