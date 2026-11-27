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
        assert 0.0 <= s["silent_neuron_frac"] <= 1.0


def test_snn_forward_with_stats_and_mem_init(tmp_path):
    """forward_with_stats matches forward(), mem_init_ratio=0.5 works, and checkpoint round-trips."""
    from models.ann import DQN
    from models.snn import SNNFromDQN
    from conversion.convert import build_snn_from_ann
    from analysis.spike_stats import compute_ann_macs_and_snn_sops
    from configs.config import FRAME_STACK

    torch.manual_seed(1)
    ann = DQN().eval()
    x = torch.rand(1, FRAME_STACK, 84, 84)
    thresholds = [1.0, 1.0, 1.0, 1.0]

    snn = build_snn_from_ann(ann, thresholds, mem_init_ratio=0.5).eval()
    with torch.no_grad():
        q_std = snn(x, T=16)
        q_stat, stats = snn.forward_with_stats(x, T=16)
    assert torch.allclose(q_std, q_stat, atol=1e-6)

    # Portable checkpoint round-trip
    ckpt_path = tmp_path / "snn_portable.pt"
    snn.save_checkpoint(ckpt_path, metadata={"percentile": 99.9})
    snn_loaded = SNNFromDQN.load_checkpoint(ckpt_path, device="cpu")
    with torch.no_grad():
        assert torch.allclose(snn_loaded(x, T=16), q_std, atol=1e-6)
    assert snn_loaded.mem_init_ratio == 0.5

    # SOPs vs MACs calculation
    layer_spikes = {k: v["total_spikes"] for k, v in stats.items() if not k.startswith("_")}
    ops = compute_ann_macs_and_snn_sops(snn, layer_spikes)
    assert ops["ann_macs_total"] > ops["ann_macs_spiking_layers"] > 0
    assert ops["conv0_analog_macs"] > 0
    assert ops["snn_sops_spiking_layers"] >= 0


def test_plot_helpers_generate_all_three_pngs(tmp_path):
    """All 3 report chart helpers write non-empty PNG files."""
    from analysis.plot_results import (
        plot_action_agreement_vs_T,
        plot_reward_vs_T,
        plot_spike_rates,
    )

    p1 = tmp_path / "reward.png"
    p2 = tmp_path / "agreement.png"
    p3 = tmp_path / "spikes.png"

    plot_reward_vs_T(90.0, -300.0, [8, 16, 32], [-50.0, 40.0, 85.0], [20.0, 15.0, 5.0], str(p1), scripted_reward=75.0)
    plot_action_agreement_vs_T([8, 16, 32], [0.5, 0.8, 0.95], str(p2))
    plot_spike_rates(["conv_0", "conv_1", "conv_2", "fc_0"], [0.12, 0.08, 0.05, 0.03], str(p3), T=32, total_spikes=4200)

    for p in (p1, p2, p3):
        assert p.exists() and p.stat().st_size > 1000
