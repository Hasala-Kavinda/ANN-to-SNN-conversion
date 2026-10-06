"""
Uniform experience replay buffer.

Stores (state, action, reward, next_state, done) tuples in pre-allocated
NumPy arrays for fast O(1) push and O(batch) sample.
"""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """
    Circular replay buffer backed by pre-allocated NumPy arrays.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    obs_shape : tuple[int, ...]
        Shape of a single observation, e.g. (4, 84, 84).
    device : str | torch.device
        Device to move sampled tensors to.
    """

    def __init__(
        self,
        capacity:  int,
        obs_shape: tuple[int, ...],
        device:    str | torch.device = "cpu",
    ) -> None:
        self.capacity  = capacity
        self.obs_shape = obs_shape
        self.device    = device

        self._states      = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._next_states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._actions     = np.zeros((capacity,), dtype=np.int64)
        self._rewards     = np.zeros((capacity,), dtype=np.float32)
        self._dones       = np.zeros((capacity,), dtype=np.float32)

        self._ptr  = 0   # next write index
        self._size = 0   # number of valid entries

    # ── Write ─────────────────────────────────────────────────────────────────

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        """Store one transition. Overwrites oldest entry when full."""
        self._states[self._ptr]      = state
        self._actions[self._ptr]     = action
        self._rewards[self._ptr]     = reward
        self._next_states[self._ptr] = next_state
        self._dones[self._ptr]       = float(done)

        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    # ── Read ──────────────────────────────────────────────────────────────────

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Uniformly sample a random minibatch.

        Returns
        -------
        (states, actions, rewards, next_states, dones)
        All torch.Tensor on self.device.
        rewards and dones have shape (B, 1) for easy broadcasting.
        """
        assert self._size >= batch_size, (
            f"Buffer has {self._size} transitions, need {batch_size}"
        )
        idx = np.random.randint(0, self._size, size=batch_size)

        def _t(arr: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(arr[idx]).to(self.device)

        return (
            _t(self._states),
            _t(self._actions),
            _t(self._rewards).unsqueeze(1),
            _t(self._next_states),
            _t(self._dones).unsqueeze(1),
        )

    def __len__(self) -> int:
        return self._size
