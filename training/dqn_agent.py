"""
Double DQN agent.

Key design choices:
  - Double DQN TD target: reduces overestimation bias.
    Action selected by online net, value from target net.
  - Hard target-network update every TARGET_UPDATE gradient steps.
  - Linear epsilon decay (EPS_START → EPS_END over EPS_DECAY env steps).
  - Huber loss (SmoothL1) for robustness to outlier rewards.
  - Gradient norm clipping for training stability.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import (
    BATCH_SIZE,
    EPS_DECAY,
    EPS_END,
    EPS_START,
    FRAME_STACK,
    GAMMA,
    GRAD_CLIP,
    IMG_H,
    IMG_W,
    LR,
    N_ACTIONS,
    REPLAY_SIZE,
    TARGET_UPDATE,
    WARMUP_STEPS,
)
from models.ann import DQN
from training.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Double DQN agent with ε-greedy exploration.

    Parameters
    ----------
    device : torch.device
        GPU or CPU for inference and training.
    """

    def __init__(self, device: torch.device) -> None:
        self.device    = device
        self.n_actions = N_ACTIONS

        # Online net (trained) + target net (periodically synced)
        self.online_net = DQN().to(device)
        self.target_net = DQN().to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = Adam(self.online_net.parameters(), lr=LR)

        self.replay = ReplayBuffer(
            capacity=REPLAY_SIZE,
            obs_shape=(FRAME_STACK, IMG_H, IMG_W),
            device=device,
        )

        self.steps_done   = 0   # environment steps
        self.updates_done = 0   # gradient updates

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        """
        ε-greedy action selection.

        Parameters
        ----------
        obs    : np.ndarray  shape (FRAME_STACK, H, W)
        greedy : bool  — skip exploration (use during evaluation)
        """
        eps = self._current_epsilon()
        if not greedy and np.random.rand() < eps:
            return int(np.random.randint(self.n_actions))

        obs_t = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            return int(self.online_net(obs_t).argmax(dim=1).item())

    def _current_epsilon(self) -> float:
        """Linear annealing: EPS_START → EPS_END over EPS_DECAY steps."""
        progress = min(self.steps_done / max(EPS_DECAY, 1), 1.0)
        return EPS_START + progress * (EPS_END - EPS_START)

    @property
    def epsilon(self) -> float:
        return self._current_epsilon()

    # ── Experience storage ────────────────────────────────────────────────────

    def store(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        """Store a transition and increment step counter."""
        self.replay.push(state, action, reward, next_state, done)
        self.steps_done += 1

    # ── Learning ──────────────────────────────────────────────────────────────

    def learn(self) -> float | None:
        """
        Sample a minibatch, compute Double-DQN loss, update online net.

        Returns loss value (float) or None if not ready to learn yet.
        """
        if (
            len(self.replay) < BATCH_SIZE
            or self.steps_done < WARMUP_STEPS
        ):
            return None

        s, a, r, ns, d = self.replay.sample(BATCH_SIZE)

        # Q(s, a) from online network
        current_q = self.online_net(s).gather(1, a.unsqueeze(1))

        # Double DQN target
        with torch.no_grad():
            next_actions = self.online_net(ns).argmax(1, keepdim=True)
            next_q  = self.target_net(ns).gather(1, next_actions)
            target_q = r + GAMMA * next_q * (1.0 - d)

        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(
                self.online_net.parameters(), GRAD_CLIP
            )
        self.optimizer.step()

        self.updates_done += 1

        # Hard target update
        if self.updates_done % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "online_net":   self.online_net.state_dict(),
                "target_net":   self.target_net.state_dict(),
                "optimizer":    self.optimizer.state_dict(),
                "steps_done":   self.steps_done,
                "updates_done": self.updates_done,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps_done   = ckpt.get("steps_done",   0)
        self.updates_done = ckpt.get("updates_done", 0)
        print(f"Loaded checkpoint: step={self.steps_done}, "
              f"updates={self.updates_done}")
