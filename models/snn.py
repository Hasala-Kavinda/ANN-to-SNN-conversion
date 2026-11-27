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

from configs.config import FRAME_STACK, IMG_H, IMG_W, N_ACTIONS


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
        conv_layers:    list[nn.Module],
        fc_layers:      list[nn.Module],
        thresholds:     list[float],
        mem_init_ratio: float = 0.0,
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

        self._n_if_conv     = n_if_conv
        self._n_if_fc       = n_if_fc
        self.thresholds     = [float(t) for t in thresholds]
        self.mem_init_ratio = float(mem_init_ratio)

    def _init_mem(self, h: torch.Tensor, threshold: float, mem_init_ratio: float) -> torch.Tensor:
        if mem_init_ratio == 0.0:
            return torch.zeros_like(h)
        return torch.full_like(h, mem_init_ratio * threshold)

    def forward(
        self,
        x: torch.Tensor,
        T: int = 32,
        mem_init_ratio: float | None = None,
    ) -> torch.Tensor:
        """
        Simulate T timesteps with constant-current input injection.

        Parameters
        ----------
        x : torch.Tensor  shape (B, C, H, W)  — same as DQN input.
        T : int           — number of simulation timesteps.
        mem_init_ratio : float | None — optional override for initial membrane
            potential as a fraction of threshold (e.g. 0.5).

        Returns
        -------
        torch.Tensor  shape (B, n_actions)
            Accumulated output membrane / T  ≈  DQN Q-values.
        """
        ratio  = self.mem_init_ratio if mem_init_ratio is None else float(mem_init_ratio)
        device = x.device
        B      = x.shape[0]

        # Initialise membrane states (lazy, allocated on first use)
        mem_conv: list[Optional[torch.Tensor]] = [None] * self._n_if_conv
        mem_fc:   list[Optional[torch.Tensor]] = [None] * self._n_if_fc

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
                    mem_conv[i] = self._init_mem(h, neuron.threshold, ratio)
                h, mem_conv[i] = neuron(h, mem_conv[i])

            # ── Flatten ──────────────────────────────────────────────────────
            h = h.flatten(1)

            # ── Hidden FC layers (all except final output) ───────────────────
            for i, (fc, neuron) in enumerate(
                zip(self.fc_layers[:-1], self.if_fc)
            ):
                h = fc(h)
                if mem_fc[i] is None:
                    mem_fc[i] = self._init_mem(h, neuron.threshold, ratio)
                h, mem_fc[i] = neuron(h, mem_fc[i])

            # ── Output layer — integrate membrane, no spike ──────────────────
            h = self.fc_layers[-1](h)
            acc_out = acc_out + h

        return acc_out / T

    def forward_with_stats(
        self,
        x: torch.Tensor,
        T: int = 32,
        mem_init_ratio: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, dict]]:
        """
        Run one T-step forward pass and collect per-layer spike statistics
        in the same pass (eliminates redundant simulation).

        Returns
        -------
        (q_values, stats)
        """
        ratio  = self.mem_init_ratio if mem_init_ratio is None else float(mem_init_ratio)
        device = x.device
        B      = x.shape[0]

        mem_conv: list[Optional[torch.Tensor]] = [None] * self._n_if_conv
        mem_fc:   list[Optional[torch.Tensor]] = [None] * self._n_if_fc
        acc_out = torch.zeros(B, self.fc_layers[-1].out_features, device=device)

        spike_counts_conv = [torch.zeros(1, device=device) for _ in range(self._n_if_conv)]
        spike_counts_fc   = [torch.zeros(1, device=device) for _ in range(self._n_if_fc)]
        fired_any_conv: list[Optional[torch.Tensor]] = [None] * self._n_if_conv
        fired_any_fc:   list[Optional[torch.Tensor]] = [None] * self._n_if_fc
        layer_sizes_conv  = [0] * self._n_if_conv
        layer_sizes_fc    = [0] * self._n_if_fc

        for _ in range(T):
            h = x
            for i, (conv, neuron) in enumerate(zip(self.conv_layers, self.if_conv)):
                h = conv(h)
                if mem_conv[i] is None:
                    mem_conv[i] = self._init_mem(h, neuron.threshold, ratio)
                    layer_sizes_conv[i] = h.numel() // B
                    fired_any_conv[i] = torch.zeros_like(h, dtype=torch.bool)
                h, mem_conv[i] = neuron(h, mem_conv[i])
                spike_counts_conv[i] = spike_counts_conv[i] + h.sum()
                fired_any_conv[i] = fired_any_conv[i] | (h > 0)

            h = h.flatten(1)
            for i, (fc, neuron) in enumerate(zip(self.fc_layers[:-1], self.if_fc)):
                h = fc(h)
                if mem_fc[i] is None:
                    mem_fc[i] = self._init_mem(h, neuron.threshold, ratio)
                    layer_sizes_fc[i] = h.numel() // B
                    fired_any_fc[i] = torch.zeros_like(h, dtype=torch.bool)
                h, mem_fc[i] = neuron(h, mem_fc[i])
                spike_counts_fc[i] = spike_counts_fc[i] + h.sum()
                fired_any_fc[i] = fired_any_fc[i] | (h > 0)

            h = self.fc_layers[-1](h)
            acc_out = acc_out + h

        layer_names = (
            [f"conv_{i}" for i in range(self._n_if_conv)]
            + [f"fc_{i}"   for i in range(self._n_if_fc)]
        )
        spike_counts = spike_counts_conv + spike_counts_fc
        fired_any    = fired_any_conv + fired_any_fc
        layer_sizes  = layer_sizes_conv + layer_sizes_fc

        stats: dict[str, dict] = {}
        total_spikes = 0.0
        for name, count, fired, size in zip(layer_names, spike_counts, fired_any, layer_sizes):
            n_spikes          = float(count.item())
            n_neurons         = size * B
            rate              = n_spikes / (n_neurons * T + 1e-9)
            temporal_sparsity = 1.0 - rate
            silent_frac       = 1.0 - (float(fired.float().sum().item()) / max(n_neurons, 1)) if fired is not None else 1.0
            stats[name] = {
                "total_spikes":       n_spikes,
                "n_neurons":          n_neurons,
                "firing_rate":        rate,
                "sparsity":           temporal_sparsity,
                "temporal_sparsity":  temporal_sparsity,
                "silent_neuron_frac": silent_frac,
            }
            total_spikes += n_spikes

        stats["_total_spikes_per_decision"] = total_spikes / B
        return acc_out / T, stats

    def get_action(self, obs: torch.Tensor, T: int = 32) -> int:
        """Greedy action (no gradient)."""
        with torch.no_grad():
            return int(self.forward(obs, T=T).argmax(dim=1).item())

    def save_checkpoint(
        self,
        path: str | Path,
        metadata: dict | None = None,
    ) -> None:
        """
        Save SNN portably as CPU state_dict + architecture metadata.
        Avoids pickling CUDA tensors or coupling to module paths.
        """
        in_channels = int(self.conv_layers[0].in_channels)
        n_actions   = int(self.fc_layers[-1].out_features)
        cpu_sd      = {k: v.detach().cpu() for k, v in self.state_dict().items()}
        payload = {
            "format":         "snn_dqn_v1",
            "state_dict":     cpu_sd,
            "in_channels":    in_channels,
            "n_actions":      n_actions,
            "img_h":          IMG_H,
            "img_w":          IMG_W,
            "thresholds":     list(self.thresholds),
            "mem_init_ratio": self.mem_init_ratio,
            "conv_specs": [
                {
                    "in_channels":  c.in_channels,
                    "out_channels": c.out_channels,
                    "kernel_size":  c.kernel_size,
                    "stride":       c.stride,
                    "padding":      c.padding,
                    "bias":         c.bias is not None,
                }
                for c in self.conv_layers
            ],
            "fc_specs": [
                {
                    "in_features":  f.in_features,
                    "out_features": f.out_features,
                    "bias":         f.bias is not None,
                }
                for f in self.fc_layers
            ],
            "metadata": metadata or {},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(
        cls,
        path:   str | Path,
        device: str | torch.device = "cpu",
    ) -> "SNNFromDQN":
        """
        Load an SNN from either the portable dict format (snn_dqn_v1)
        or a legacy pickled SNNFromDQN instance.
        """
        obj = torch.load(path, map_location=device, weights_only=False)
        if isinstance(obj, cls):
            obj.to(device)
            obj.eval()
            return obj

        if isinstance(obj, dict) and obj.get("format") == "snn_dqn_v1":
            in_ch = int(obj["in_channels"])
            if in_ch != FRAME_STACK:
                raise ValueError(
                    f"SNN checkpoint expects in_channels={in_ch}, "
                    f"but configs.config.FRAME_STACK={FRAME_STACK}."
                )
            conv_layers = [
                nn.Conv2d(
                    s["in_channels"], s["out_channels"],
                    kernel_size=s["kernel_size"], stride=s["stride"],
                    padding=s["padding"], bias=s["bias"],
                )
                for s in obj["conv_specs"]
            ]
            fc_layers = [
                nn.Linear(s["in_features"], s["out_features"], bias=s["bias"])
                for s in obj["fc_specs"]
            ]
            snn = cls(
                conv_layers=conv_layers,
                fc_layers=fc_layers,
                thresholds=obj["thresholds"],
                mem_init_ratio=obj.get("mem_init_ratio", 0.0),
            )
            snn.load_state_dict(obj["state_dict"])
            snn.to(device)
            snn.eval()
            return snn

        raise ValueError(f"Unrecognized SNN checkpoint format at {path}")


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
