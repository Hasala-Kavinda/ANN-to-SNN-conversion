"""
VizDoom environment wrapper for the 'basic' scenario.

The 'basic' scenario:
  - A monster appears at a random horizontal position.
  - The agent must turn left/right to aim and then shoot.
  - Actions: MOVE_LEFT, MOVE_RIGHT, ATTACK  (3 discrete actions)
  - Rewards: +101 for a kill, -5 for a missed shot, -1 per step.

Usage:
    env = DoomEnv()
    obs = env.reset()            # np.ndarray, shape (FRAME_STACK, H, W)
    obs, reward, done = env.step(action_idx)
    env.close()
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import vizdoom as vzd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import (
    FRAME_SKIP,
    FRAME_STACK,
    IMG_H,
    IMG_W,
    N_ACTIONS,
    WINDOW_VISIBLE,
)

# Built-in VizDoom scenario directory
_VZD_SCENARIOS = Path(vzd.__file__).parent / "scenarios"


class DoomEnv:
    """
    Thin wrapper around VizDoom for the basic scenario.

    Parameters
    ----------
    scenario : str
        Name of the scenario (without extension).
    frame_skip : int
        Number of frames the chosen action is repeated for.
        Rewards from all skipped frames are accumulated.
    frame_stack : int
        Number of consecutive preprocessed frames stacked along axis 0.
    seed : int | None
        Random seed passed to VizDoom.
    window_visible : bool
        Show the game window. Always False on headless servers.
    """

    # 3 discrete actions for the basic scenario
    ACTIONS: list[list[bool]] = [
        [True,  False, False],   # 0 — MOVE_LEFT
        [False, True,  False],   # 1 — MOVE_RIGHT
        [False, False, True],    # 2 — ATTACK
    ]

    def __init__(
        self,
        scenario: str = "basic",
        frame_skip: int = FRAME_SKIP,
        frame_stack: int = FRAME_STACK,
        seed: Optional[int] = None,
        window_visible: bool = WINDOW_VISIBLE,
    ) -> None:
        self.frame_skip  = frame_skip
        self.frame_stack = frame_stack
        self._stack: list[np.ndarray] = []

        cfg_path = _VZD_SCENARIOS / f"{scenario}.cfg"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"Scenario config not found: {cfg_path}\n"
                f"Available: {list(_VZD_SCENARIOS.glob('*.cfg'))}"
            )

        self.game = vzd.DoomGame()
        self.game.load_config(str(cfg_path))

        # Force grayscale — fewer channels, simpler SNN
        self.game.set_screen_format(vzd.ScreenFormat.GRAY8)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        self.game.set_window_visible(window_visible)

        if seed is not None:
            self.game.set_seed(seed)

        self.game.init()

        self._episode_steps  = 0
        self._episode_reward = 0.0

    # ── Public API ───────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Start a new episode. Returns the initial stacked observation."""
        self.game.new_episode()
        self._episode_steps  = 0
        self._episode_reward = 0.0

        frame = self._preprocess(self._get_raw_frame())
        self._stack = [frame.copy() for _ in range(self.frame_stack)]
        return self._get_obs()

    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool]:
        """
        Apply action for `frame_skip` frames.

        Parameters
        ----------
        action_idx : int
            Index into DoomEnv.ACTIONS (0=left, 1=right, 2=attack).

        Returns
        -------
        obs : np.ndarray, shape (frame_stack, H, W), dtype float32
        reward : float
        done : bool
        """
        assert 0 <= action_idx < N_ACTIONS, f"Invalid action index: {action_idx}"

        reward = self.game.make_action(self.ACTIONS[action_idx], self.frame_skip)
        done   = self.game.is_episode_finished()

        frame = self._preprocess(
            self._get_raw_frame() if not done else None
        )
        self._stack.pop(0)
        self._stack.append(frame)

        self._episode_steps  += 1
        self._episode_reward += reward

        return self._get_obs(), float(reward), done

    def close(self) -> None:
        """Shut down the VizDoom instance."""
        self.game.close()

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    @property
    def obs_shape(self) -> tuple[int, int, int]:
        """(frame_stack, IMG_H, IMG_W)"""
        return (self.frame_stack, IMG_H, IMG_W)

    @property
    def episode_reward(self) -> float:
        return self._episode_reward

    @property
    def episode_steps(self) -> int:
        return self._episode_steps

    # ── Internals ────────────────────────────────────────────────────────────

    def _get_raw_frame(self) -> Optional[np.ndarray]:
        state = self.game.get_state()
        return state.screen_buffer if state is not None else None

    def _preprocess(self, frame: Optional[np.ndarray]) -> np.ndarray:
        """
        Raw GRAY8 frame → cropped → resized (IMG_H×IMG_W) → float32 [0,1].

        Cropping the top 40 rows removes ceiling/sky that carries no
        useful information for aiming, giving more resolution to the
        monster region.
        """
        if frame is None:
            return np.zeros((IMG_H, IMG_W), dtype=np.float32)
        frame = frame[40:, :]   # crop top 40 px
        frame = cv2.resize(frame, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        return frame.astype(np.float32) / 255.0

    def _get_obs(self) -> np.ndarray:
        """Stack frames along axis 0 → (frame_stack, H, W)."""
        return np.stack(self._stack, axis=0)

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ── Smoke test ───────────────────────────────────────────────────────────────

def _smoke_test(n_episodes: int = 5) -> None:
    import random, time
    print(f"Running {n_episodes} random episodes as smoke test…")
    env = DoomEnv()
    rewards = []
    for ep in range(n_episodes):
        obs  = env.reset()
        assert obs.shape == env.obs_shape, f"Bad obs shape: {obs.shape}"
        done = False
        t0   = time.time()
        while not done:
            obs, _, done = env.step(random.randint(0, env.n_actions - 1))
        elapsed = time.time() - t0
        print(f"  ep {ep+1:2d}: reward={env.episode_reward:7.1f}  "
              f"steps={env.episode_steps:4d}  time={elapsed:.2f}s")
        rewards.append(env.episode_reward)
    env.close()
    print(f"\nMean reward over {n_episodes} episodes: {sum(rewards)/len(rewards):.2f}")
    print("Smoke test PASSED ✓")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=5)
    _smoke_test(p.parse_args().episodes)
