"""
MNIST ANN-to-SNN sanity check (Step 6).

Trains a small conversion-friendly CNN on MNIST, converts it to an SNN
using threshold normalization, and reports accuracy at several T values.

This script validates the spiking machinery in isolation before applying
it to the DOOM DQN.

Expected results: SNN accuracy within ~1% of ANN accuracy at T >= 32.

Usage
-----
python mnist_sanity/mnist_snn_test.py
python mnist_sanity/mnist_snn_test.py --T 16 32 64 128
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Conversion-friendly CNN ───────────────────────────────────────────────────

class MnistCNN(nn.Module):
    """Small ANN for MNIST. ReLU only, avg pool, no BN/dropout."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=1, padding=2),  # 28×28
            nn.ReLU(),
            nn.AvgPool2d(2),                            # 14×14
            nn.Conv2d(16, 32, 5, stride=1, padding=2), # 14×14
            nn.ReLU(),
            nn.AvgPool2d(2),                            # 7×7
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).flatten(1))


# ── IF SNN ────────────────────────────────────────────────────────────────────

class IFNeuron(nn.Module):
    def __init__(self, threshold: float = 1.0) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, current, membrane):
        membrane = membrane + current
        spikes   = (membrane >= self.threshold).float()
        membrane = membrane - spikes * self.threshold
        return spikes, membrane


class MnistSNN(nn.Module):
    """
    SNN mirror of MnistCNN, built from converted weights and thresholds.
    """

    def __init__(
        self,
        conv_layers: list[nn.Module],
        pool_layers: list[nn.Module],
        fc_layers:   list[nn.Module],
        thresholds:  list[float],
    ) -> None:
        super().__init__()
        self.conv_layers = nn.ModuleList(conv_layers)
        self.pool_layers = nn.ModuleList(pool_layers)
        self.fc_layers   = nn.ModuleList(fc_layers)

        # One IF neuron per ReLU: 2 conv + 1 hidden fc
        assert len(thresholds) == 3
        self.if_conv = nn.ModuleList([IFNeuron(t) for t in thresholds[:2]])
        self.if_fc   = IFNeuron(thresholds[2])

    def forward(self, x: torch.Tensor, T: int = 32) -> torch.Tensor:
        B      = x.shape[0]
        device = x.device

        mem_c = [None, None]
        mem_f = None
        acc   = torch.zeros(B, 10, device=device)

        for _ in range(T):
            h = x
            # Conv + pool stack
            for i, (conv, pool, neuron) in enumerate(
                zip(self.conv_layers, self.pool_layers, self.if_conv)
            ):
                h = conv(h)
                if mem_c[i] is None:
                    mem_c[i] = torch.zeros_like(h)
                h, mem_c[i] = neuron(h, mem_c[i])
                h = pool(h)

            h = h.flatten(1)

            # Hidden FC
            h = self.fc_layers[0](h)
            if mem_f is None:
                mem_f = torch.zeros_like(h)
            h, mem_f = self.if_fc(h, mem_f)

            # Output FC — read membrane directly
            h   = self.fc_layers[1](h)
            acc = acc + h

        return acc / T


# ── Calibration + conversion ──────────────────────────────────────────────────

def calibrate_and_convert(
    ann:         MnistCNN,
    loader:      DataLoader,
    percentile:  float = 99.9,
    n_batches:   int   = 20,
    device:      torch.device = torch.device("cpu"),
) -> MnistSNN:
    """
    Collect activations from each ReLU position and compute thresholds,
    then build and return the converted SNN.
    """
    ann.eval()
    activations: list[list[torch.Tensor]] = [[], [], []]   # 3 ReLU layers

    hooks = []

    def make_hook(idx):
        def hook(module, inp, out):
            activations[idx].append(out.detach().cpu().flatten())
        return hook

    # Register hooks on each ReLU
    relu_modules = [m for m in ann.modules() if isinstance(m, nn.ReLU)]
    for i, relu in enumerate(relu_modules[:3]):
        hooks.append(relu.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            if i >= n_batches:
                break
            ann(imgs.to(device))

    for h in hooks:
        h.remove()

    thresholds = []
    for i, acts in enumerate(activations):
        combined = torch.cat(acts).numpy()
        thresh   = float(np.percentile(combined, percentile))
        thresh   = max(thresh, 1e-6)   # guard against zero
        thresholds.append(thresh)
        print(f"  Layer {i} threshold ({percentile}th pctile): {thresh:.4f}")

    # Extract and scale layers
    conv_layers = []
    pool_layers = []
    fc_layers   = []

    # Scale weights: W_snn = W_ann * (thresh_pre / thresh_post)
    # For input layer: input is in [0,1] so we treat thresh_in = 1.0
    # Pattern: W_scaled[i] = W[i] * (thresh[i-1] / thresh[i])
    # Input scale = 1.0 (image pixels already in [0,1])
    scales = [1.0] + thresholds  # scales[i] is the "pre-layer" threshold

    # Conv layers with weight scaling
    src_convs = [ann.conv[0], ann.conv[3]]
    for i, src in enumerate(src_convs):
        new_conv = nn.Conv2d(
            src.in_channels, src.out_channels,
            src.kernel_size, src.stride, src.padding, bias=True
        )
        scale = scales[i] / thresholds[i]
        with torch.no_grad():
            new_conv.weight.copy_(src.weight * scale)
            if src.bias is not None:
                new_conv.bias.copy_(src.bias * scale)
        conv_layers.append(new_conv)
        pool_layers.append(nn.AvgPool2d(2))

    # FC layers
    src_fcs = [ann.fc[0], ann.fc[2]]
    fc_scales = [
        scales[2] / thresholds[2],  # hidden fc
        1.0 / thresholds[2],         # output fc (read membrane, no thresh)
    ]
    for i, src in enumerate(src_fcs):
        new_fc = nn.Linear(src.in_features, src.out_features, bias=True)
        scale  = fc_scales[i]
        with torch.no_grad():
            new_fc.weight.copy_(src.weight * scale)
            if src.bias is not None:
                new_fc.bias.copy_(src.bias * scale)
        fc_layers.append(new_fc)

    return MnistSNN(conv_layers, pool_layers, fc_layers, thresholds)


# ── Evaluation helpers ────────────────────────────────────────────────────────

@torch.no_grad()
def eval_ann(model: MnistCNN, loader: DataLoader, device) -> float:
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total


@torch.no_grad()
def eval_snn(model: MnistSNN, loader: DataLoader, device, T: int) -> float:
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs, T=T).argmax(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total


# ── Training ──────────────────────────────────────────────────────────────────

def train_mnist_ann(
    device: torch.device,
    epochs: int = 5,
    batch:  int = 128,
) -> tuple[MnistCNN, DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST("data/mnist", train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST("data/mnist", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=batch, shuffle=False, num_workers=4)

    ann = MnistCNN().to(device)
    opt = torch.optim.Adam(ann.parameters(), lr=1e-3)

    print("Training MNIST ANN …")
    for ep in range(1, epochs + 1):
        ann.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            loss = F.cross_entropy(ann(imgs), labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        acc = eval_ann(ann, test_loader, device)
        print(f"  Epoch {ep}/{epochs}  loss={total_loss/len(train_loader):.4f}  "
              f"test_acc={acc*100:.2f}%")

    return ann, train_loader, test_loader


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--T",      type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch",  type=int, default=128)
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    ann, train_loader, test_loader = train_mnist_ann(
        device, epochs=args.epochs, batch=args.batch
    )

    ann_acc = eval_ann(ann, test_loader, device)
    print(f"\nANN test accuracy: {ann_acc*100:.2f}%")

    print("\nCalibrating thresholds …")
    snn = calibrate_and_convert(ann, train_loader, percentile=99.9, device=device)
    snn = snn.to(device)

    print("\nSNN accuracy sweep:")
    print(f"  {'T':>6}  {'SNN acc':>10}  {'Gap vs ANN':>12}")
    print("  " + "-" * 32)
    for T in args.T:
        snn_acc = eval_snn(snn, test_loader, device, T=T)
        gap     = ann_acc - snn_acc
        marker  = "✓" if abs(gap) < 0.01 else "✗"
        print(f"  {T:>6}  {snn_acc*100:>9.2f}%  {gap*100:>+10.2f}%  {marker}")

    print("\nTarget: SNN within 1% of ANN at T >= 32")
    print("MNIST sanity check complete.")


if __name__ == "__main__":
    main()
