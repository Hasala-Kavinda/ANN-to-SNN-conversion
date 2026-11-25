"""
Unit tests for DoomEnv.
Run with  pytest tests/test_env.py -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_env_import():
    from envs.doom_env import DoomEnv
    assert DoomEnv is not None


def test_env_reset_shape():
    from envs.doom_env import DoomEnv
    from configs.config import FRAME_STACK, IMG_H, IMG_W
    env = DoomEnv()
    obs = env.reset()
    assert obs.shape == (FRAME_STACK, IMG_H, IMG_W), (
        f"Expected ({FRAME_STACK}, {IMG_H}, {IMG_W}), got {obs.shape}"
    )
    assert obs.dtype == np.float32
    assert 0.0 <= obs.min() and obs.max() <= 1.0
    env.close()


def test_env_step_returns():
    from envs.doom_env import DoomEnv
    from configs.config import FRAME_STACK, IMG_H, IMG_W
    env = DoomEnv()
    env.reset()
    obs, reward, done = env.step(0)
    assert obs.shape == (FRAME_STACK, IMG_H, IMG_W)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    env.close()


def test_env_invalid_action():
    from envs.doom_env import DoomEnv
    env = DoomEnv()
    env.reset()
    with pytest.raises(AssertionError):
        env.step(99)
    env.close()


def test_env_context_manager():
    from envs.doom_env import DoomEnv
    with DoomEnv() as env:
        obs = env.reset()
        assert obs is not None


def test_env_full_episode():
    """One complete episode runs without errors."""
    from envs.doom_env import DoomEnv
    import random
    env  = DoomEnv()
    _    = env.reset()
    done = False
    steps = 0
    while not done and steps < 2000:
        _, _, done = env.step(random.randint(0, env.n_actions - 1))
        steps += 1
    env.close()
    assert steps > 0


def test_100_episodes():
    """Checkpoint: 100 random episodes without errors."""
    from envs.doom_env import DoomEnv
    import random
    env = DoomEnv()
    for _ in range(100):
        env.reset()
        done = False
        while not done:
            _, _, done = env.step(random.randint(0, env.n_actions - 1))
    env.close()
