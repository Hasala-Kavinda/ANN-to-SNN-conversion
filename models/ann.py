"""
Conversion-friendly DQN architecture for VizDoom.

Design constraints for ANN-to-SNN conversion:
  - ReLU only (no sigmoid, softmax, tanh, layernorm, batchnorm, dropout)
  - Strided convolutions instead of MaxPool (MaxPool's winner-take-all
    semantics don't map cleanly to spike rates)
  - No weight normalization layers
  - Biases kept on (absorbed into thresholds at conversion time)

Architecture (input: B × FRAME_STACK × 84 × 84):
  Conv2d(C,  32, 8, stride=4) → ReLU  →  (B, 32, 20, 20)
  Conv2d(32, 64, 4, stride=2) → ReLU  →  (B, 64,  9,  9)
  Conv2d(64, 64, 3, stride=1) → ReLU  →  (B, 64,  7,  7)
  Flatten                               →  (B, 3136)
  Linear(3136, 512)           → ReLU
  Linear(512,  3)                        (raw Q-values, no activation)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import FRAME_STACK, IMG_H, IMG_W, N_ACTIONS


class DQN(nn.Module):
    """
    Conversion-friendly Deep Q-Network for VizDoom basic.

    Parameters
    ----------
    in_channels : int
        Number of stacked input frames (= FRAME_STACK).
    n_actions : int
        Number of discrete actions (3 for VizDoom basic).
    """

    def __init__(
        self,
        in_channels: int = FRAME_STACK,
        n_actions: int = N_ACTIONS,
    ) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=True),
            nn.ReLU(),
        )

        # Compute flattened size dynamically so it works for any IMG_H/W
        self._flat_size = self._compute_flat_size(in_channels)

        self.fc = nn.Sequential(
            nn.Linear(self._flat_size, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, n_actions, bias=True),
            # No activation — raw Q-value outputs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (B, in_channels, 84, 84)  float32 in [0,1]

        Returns
        -------
        torch.Tensor  shape (B, n_actions)
        """
        return self.fc(self.conv(x).flatten(1))

    def _compute_flat_size(self, in_channels: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, IMG_H, IMG_W)
            return int(self.conv(dummy).flatten(1).shape[1])

    def get_action(self, obs: torch.Tensor) -> int:
        """Greedy action — no gradient tracked."""
        with torch.no_grad():
            return int(self.forward(obs).argmax(dim=1).item())

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "DQN":
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model


# ── Shape test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    net = DQN()
    x   = torch.zeros(1, FRAME_STACK, 84, 84)
    out = net(x)
    print(f"Input  shape : {tuple(x.shape)}")
    print(f"Output shape : {tuple(out.shape)}  (expected (1, {N_ACTIONS}))")
    assert out.shape == (1, N_ACTIONS), f"Shape mismatch: {out.shape}"
    total = sum(p.numel() for p in net.parameters())
    print(f"Total params : {total:,}")
    print("ANN shape test PASSED ✓")
