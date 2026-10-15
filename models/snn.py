"""
Spiking Neural Network (SNN) built from a converted DQN.

Neuron model: Integrate-and-Fire (IF) with reset-by-subtraction.
  - membrane += input_current
  - fire (spike=1) when membrane >= threshold
  - subtract threshold on fire — preserves the fractional charge
    so the mapping to ReLU activations is exact at infinite T.

Why reset-by-subtraction (not reset-to-zero)?
  A ReLU output of v for a layer with threshold λ maps to a neuron
  that fires exactly ⌊v/λ⌋ spikes at finite T, carrying the remainder
  into the next timestep. Reset-to-zero discards the remainder, causing
  a systematic downward bias in Q-value estimates.

Output readout:
  The final Linear layer has no IF neuron. We accumulate its output
  into a running membrane sum over T timesteps, then divide by T.
  This recovers a continuous Q-value estimate far more precisely than
  counting output spikes.

Input encoding:
  Constant-current (direct) injection: the same preprocessed frame is
  fed as input current at every timestep. This avoids the long latency
  of Poisson rate coding while retaining the efficiency of sparsity
  in hidden layers.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import FRAME_STACK, N_ACTIONS


# ── Neuron primitive ──────────────────────────────────────────────────────────

class IFNeuron(nn.Module):
    """
    Reset-by-subtraction Integrate-and-Fire neuron layer (stateless call).

    The caller is responsible for maintaining and passing the membrane
    tensor between timesteps.

    Parameters
    ----------
    threshold : float
        Firing threshold Vth.
    """

    def __init__(self, threshold: float = 1.0) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(
        self,
        current: torch.Tensor,
        membrane: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        current  : input current at this timestep (any shape).
        membrane : membrane potential from previous timestep (same shape).

        Returns
        -------
        spikes   : float tensor of 0s and 1s, same shape.
        membrane : updated membrane potential.
        """
        membrane = membrane + current
        spikes   = (membrane >= self.threshold).float()
        membrane = membrane - spikes * self.threshold   # reset-by-subtraction
        return spikes, membrane


# ── Full SNN ──────────────────────────────────────────────────────────────────

class SNNFromDQN(nn.Module):
    """
    SNN equivalent of the DQN, built from converted (weight-scaled) layers.

    The network mirrors the DQN's conv→fc structure. Every ReLU position
    is replaced by an IFNeuron with a calibrated threshold. The last
    Linear layer (output) has no neuron — we read its accumulated
    membrane potential directly.

    Parameters
    ----------
    conv_layers : list[nn.Conv2d]
        Extracted and weight-scaled Conv2d layers (len=3).
    fc_layers   : list[nn.Linear]
        Extracted and weight-scaled Linear layers (len=2).
        fc_layers[-1] is the output layer (no IF neuron).
    thresholds  : list[float]
        One threshold per ReLU position, in order:
        [conv1_thresh, conv2_thresh, conv3_thresh, fc1_thresh]
        (4 values for the 3-conv + 1-hidden-fc architecture).
    """

    def __init__(
        self,
        conv_layers: list[nn.Module],
        fc_layers:   list[nn.Module],
        thresholds:  list[float],
    ) -> None:
        super().__init__()

        n_if_conv = len(conv_layers)        # 3
        n_if_fc   = len(fc_layers) - 1     # 1  (last layer = readout)
        n_thresh  = n_if_conv + n_if_fc    # 4

        assert len(thresholds) == n_thresh, (
            f"Expected {n_thresh} thresholds (conv×{n_if_conv} + fc×{n_if_fc}), "
            f"got {len(thresholds)}"
        )

        self.conv_layers = nn.ModuleList(conv_layers)
        self.fc_layers   = nn.ModuleList(fc_layers)

        self.if_conv = nn.ModuleList(
            [IFNeuron(t) for t in thresholds[:n_if_conv]]
        )
        self.if_fc = nn.ModuleList(
            [IFNeuron(t) for t in thresholds[n_if_conv:]]
        )

        self._n_if_conv = n_if_conv
        self._n_if_fc   = n_if_fc

    def forward(self, x: torch.Tensor, T: int = 32) -> torch.Tensor:
        """
        Simulate T timesteps with constant-current input injection.

        Parameters
        ----------
        x : torch.Tensor  shape (B, C, H, W)  — same as DQN input.
        T : int           — number of simulation timesteps.

        Returns
        -------
        torch.Tensor  shape (B, n_actions)
            Accumulated output membrane / T  ≈  DQN Q-values.
        """
        device = x.device
        B      = x.shape[0]

        # Initialise membrane states (lazy, allocated on first use)
        mem_conv: list[Optional[torch.Tensor]] = [None] * self._n_if_conv
        mem_fc:   list[Optional[torch.Tensor]] = [None] * self._n_if_fc
        mem_out:  Optional[torch.Tensor]       = None

        acc_out = torch.zeros(
            B, self.fc_layers[-1].out_features, device=device
        )

        for _ in range(T):
            h = x   # constant-current: same frame injected every tick

            # ── Conv stack ───────────────────────────────────────────────────
            for i, (conv, neuron) in enumerate(
                zip(self.conv_layers, self.if_conv)
            ):
                h = conv(h)
                if mem_conv[i] is None:
                    mem_conv[i] = torch.zeros_like(h)
                h, mem_conv[i] = neuron(h, mem_conv[i])

            # ── Flatten ──────────────────────────────────────────────────────
            h = h.flatten(1)

            # ── Hidden FC layers (all except final output) ───────────────────
            for i, (fc, neuron) in enumerate(
                zip(self.fc_layers[:-1], self.if_fc)
            ):
                h = fc(h)
                if mem_fc[i] is None:
                    mem_fc[i] = torch.zeros_like(h)
                h, mem_fc[i] = neuron(h, mem_fc[i])

            # ── Output layer — integrate membrane, no spike ──────────────────
            h = self.fc_layers[-1](h)
            if mem_out is None:
                mem_out = torch.zeros_like(h)
            mem_out  = mem_out + h
            acc_out  = acc_out + mem_out

        return acc_out / T

    def get_action(self, obs: torch.Tensor, T: int = 32) -> int:
        """Greedy action (no gradient)."""
        with torch.no_grad():
            return int(self.forward(obs, T=T).argmax(dim=1).item())


# ── Shape test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from models.ann import DQN

    ann = DQN()
    # Extract layers (no weight scaling — pure shape test)
    conv_layers = [ann.conv[0], ann.conv[2], ann.conv[4]]   # skip ReLUs
    fc_layers   = [ann.fc[0],  ann.fc[2]]                   # skip ReLUs
    thresholds  = [1.0, 1.0, 1.0, 1.0]

    snn = SNNFromDQN(conv_layers, fc_layers, thresholds)
    x   = torch.zeros(1, FRAME_STACK, 84, 84)
    out = snn(x, T=16)
    print(f"SNN output shape: {tuple(out.shape)}  (expected (1, {N_ACTIONS}))")
    assert out.shape == (1, N_ACTIONS), f"Shape mismatch: {out.shape}"
    print("SNN shape test PASSED ✓")
