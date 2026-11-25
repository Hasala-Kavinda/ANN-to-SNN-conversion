"""
Unit tests for ANN and SNN architecture.
Run with:  pytest tests/test_models.py -v
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_ann_output_shape():
    from models.ann import DQN
    from configs.config import FRAME_STACK, N_ACTIONS
    net = DQN()
    x   = torch.zeros(1, FRAME_STACK, 84, 84)
    out = net(x)
    assert out.shape == (1, N_ACTIONS), f"Got {out.shape}"


def test_ann_batch():
    from models.ann import DQN
    from configs.config import FRAME_STACK, N_ACTIONS
    net = DQN()
    x   = torch.zeros(8, FRAME_STACK, 84, 84)
    out = net(x)
    assert out.shape == (8, N_ACTIONS)


def test_ann_save_load(tmp_path):
    from models.ann import DQN
    from configs.config import FRAME_STACK, N_ACTIONS
    net  = DQN()
    path = tmp_path / "test.pt"
    net.save(path)
    net2 = DQN.load(path)
    x = torch.zeros(1, FRAME_STACK, 84, 84)
    assert torch.allclose(net(x), net2(x))


def test_snn_output_shape():
    from models.ann import DQN
    from models.snn import SNNFromDQN
    from configs.config import FRAME_STACK, N_ACTIONS

    ann = DQN()
    conv_layers = [ann.conv[0], ann.conv[2], ann.conv[4]]
    fc_layers   = [ann.fc[0],  ann.fc[2]]
    thresholds  = [1.0, 1.0, 1.0, 1.0]

    snn = SNNFromDQN(conv_layers, fc_layers, thresholds)
    x   = torch.zeros(1, FRAME_STACK, 84, 84)
    out = snn(x, T=8)
    assert out.shape == (1, N_ACTIONS), f"Got {out.shape}"


def test_snn_T_sweep():
    """SNN output shape is independent of T."""
    from models.ann import DQN
    from models.snn import SNNFromDQN
    from configs.config import FRAME_STACK, N_ACTIONS

    ann = DQN()
    conv_layers = [ann.conv[0], ann.conv[2], ann.conv[4]]
    fc_layers   = [ann.fc[0],  ann.fc[2]]
    thresholds  = [1.0, 1.0, 1.0, 1.0]
    snn = SNNFromDQN(conv_layers, fc_layers, thresholds)
    x   = torch.zeros(1, FRAME_STACK, 84, 84)

    for T in [8, 16, 32, 64]:
        out = snn(x, T=T)
        assert out.shape == (1, N_ACTIONS)


def test_replay_buffer():
    from training.replay_buffer import ReplayBuffer
    import numpy as np

    buf = ReplayBuffer(capacity=100, obs_shape=(4, 84, 84))
    obs = np.zeros((4, 84, 84), dtype=np.float32)

    for i in range(10):
        buf.push(obs, i % 3, float(i), obs, i == 9)

    assert len(buf) == 10
    s, a, r, ns, d = buf.sample(4)
    assert s.shape  == (4, 4, 84, 84)
    assert a.shape  == (4,)
    assert r.shape  == (4, 1)
    assert ns.shape == (4, 4, 84, 84)
    assert d.shape  == (4, 1)


def test_snn_conversion_fidelity_and_spike_stats():
    """Converted SNN Q-values converge toward ANN Q-values and spike stats stay in [0, 1]."""
    from models.ann import DQN
    from conversion.convert import build_snn_from_ann
    from analysis.spike_stats import collect_spike_stats
    from configs.config import FRAME_STACK

    torch.manual_seed(0)
    ann = DQN().eval()
    x = torch.rand(2, FRAME_STACK, 84, 84)

    # Collect per-ReLU max activations on x for exact calibration
    thresholds = []
    hooks = []
    for m in ann.modules():
        if isinstance(m, torch.nn.ReLU):
            hooks.append(
                m.register_forward_hook(
                    lambda _mod, _inp, out: thresholds.append(max(float(out.max().item()), 1e-6))
                )
            )
    with torch.no_grad():
        q_ann = ann(x)
    for h in hooks:
        h.remove()

    snn = build_snn_from_ann(ann, thresholds).eval()
    with torch.no_grad():
        q_snn = snn(x, T=256)

    # Converted SNN should approximate ANN Q-values closely at T=256
    assert torch.allclose(q_snn, q_ann, atol=0.05), f"ANN={q_ann}, SNN={q_snn}"

    # Spike stats firing rates and sparsity must be valid probabilities in [0, 1]
    stats = collect_spike_stats(snn, x[:1], T=32)
    for name, s in stats.items():
        if name.startswith("_"):
            continue
        assert 0.0 <= s["firing_rate"] <= 1.0, f"{name} bad rate: {s['firing_rate']}"
        assert 0.0 <= s["sparsity"] <= 1.0, f"{name} bad sparsity: {s['sparsity']}"
