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


def test_replay_buffer_uint8_roundtrip():
    """ReplayBuffer stores uint8 internally and reconstructs k/255 frames bit-identically."""
    import numpy as np
    from training.replay_buffer import ReplayBuffer

    buf = ReplayBuffer(capacity=16, obs_shape=(4, 84, 84))
    assert buf._states.dtype == np.uint8
    assert buf._next_states.dtype == np.uint8

    rng = np.random.default_rng(7)
    u8_frames = rng.integers(0, 256, size=(4, 84, 84), dtype=np.uint8)
    f32_obs = u8_frames.astype(np.float32) / 255.0
    next_u8 = rng.integers(0, 256, size=(4, 84, 84), dtype=np.uint8)
    next_f32 = next_u8.astype(np.float32) / 255.0

    buf.push(f32_obs, 1, 10.0, next_f32, False)
    s, a, r, ns, d = buf.sample(1)
    assert torch.equal(s[0], torch.from_numpy(f32_obs))
    assert torch.equal(ns[0], torch.from_numpy(next_f32))
    assert int(a[0].item()) == 1


def test_snn_conv1_caching_matches_reference():
    """SNN forward() with conv1 caching matches a reference loop calling conv1 every timestep."""
    from models.ann import DQN
    from conversion.convert import build_snn_from_ann
    from configs.config import FRAME_STACK

    torch.manual_seed(12)
    ann = DQN().eval()
    x = torch.rand(2, FRAME_STACK, 84, 84)
    thresholds = [0.5, 0.4, 0.3, 1.2]

    def _ref_forward(snn, inp, T, ratio):
        B = inp.shape[0]
        mem_c = [None] * snn._n_if_conv
        mem_f = [None] * snn._n_if_fc
        acc = torch.zeros(B, snn.fc_layers[-1].out_features, device=inp.device)
        for _ in range(T):
            h = inp
            for i, (conv, neuron) in enumerate(zip(snn.conv_layers, snn.if_conv)):
                h = conv(h)
                if mem_c[i] is None:
                    mem_c[i] = snn._init_mem(h, neuron.threshold, ratio)
                h, mem_c[i] = neuron(h, mem_c[i])
            h = h.flatten(1)
            for i, (fc, neuron) in enumerate(zip(snn.fc_layers[:-1], snn.if_fc)):
                h = fc(h)
                if mem_f[i] is None:
                    mem_f[i] = snn._init_mem(h, neuron.threshold, ratio)
                h, mem_f[i] = neuron(h, mem_f[i])
            acc = acc + snn.fc_layers[-1](h)
        return acc / T

    for mem_init in (0.0, 0.5):
        snn = build_snn_from_ann(ann, thresholds, mem_init_ratio=mem_init).eval()
        for T in (8, 32):
            with torch.no_grad():
                q_cached = snn(x, T=T)
                q_ref = _ref_forward(snn, x, T=T, ratio=mem_init)
            assert torch.allclose(q_cached, q_ref, atol=1e-6)


def test_compute_thresholds_per_layer_and_clip():
    """compute_thresholds supports per-layer percentiles, reports clip fracs, and validates length."""
    import numpy as np
    import pytest
    from conversion.normalize import compute_thresholds

    rng = np.random.default_rng(0)
    stats = {f"relu_{i}": rng.uniform(0.0, 10.0, size=10_000).astype(np.float32) for i in range(4)}
    pcts = [90.0, 95.0, 99.0, 100.0]

    thresh, clips = compute_thresholds(stats, pcts, verbose=False, return_clip_fracs=True)
    assert list(thresh.keys()) == ["relu_0", "relu_1", "relu_2", "relu_3"]
    for i, pct in enumerate(pcts):
        expected_clip = (100.0 - pct) / 100.0
        assert abs(clips[f"relu_{i}"] - expected_clip) < 0.01

    with pytest.raises(AssertionError):
        compute_thresholds(stats, [99.0, 99.5], verbose=False)


def test_sweep_stage1_tiny():
    """screen_offline returns valid metrics for every (percentile, mem_init, T) combination."""
    import numpy as np
    from models.ann import DQN
    from conversion.sweep import screen_offline
    from configs.config import FRAME_STACK

    torch.manual_seed(3)
    ann = DQN().eval()
    rng = np.random.default_rng(3)
    stats = {f"relu_{i}": rng.uniform(0.0, 2.0, size=1000).astype(np.float32) for i in range(4)}
    frames_u8 = rng.integers(0, 256, size=(16, FRAME_STACK, 84, 84), dtype=np.uint8)

    rows = screen_offline(
        ann, stats, frames_u8,
        percentiles=[99.0, 100.0],
        mem_inits=[0.0, 0.5],
        T_values=[4, 8],
        device=torch.device("cpu"),
        batch_size=8,
    )
    assert len(rows) == 2 * 2 * 2
    for r in rows:
        assert 0.0 <= r["agreement"] <= 1.0
        assert r["regret"] >= -1e-6
        assert r["q_mae"] >= 0.0
        assert 0.0 <= r["dead_neuron_frac"] <= 1.0


def test_bootstrap_and_paired_stats():
    """bootstrap_ci and paired_summary produce valid bounds and handle identical arrays."""
    import numpy as np
    from analysis.stats import bootstrap_ci, paired_summary

    rng = np.random.default_rng(42)
    x_small = rng.normal(50.0, 10.0, size=20)
    x_large = rng.normal(50.0, 10.0, size=200)
    lo_s, hi_s = bootstrap_ci(x_small, seed=0)
    lo_l, hi_l = bootstrap_ci(x_large, seed=0)
    assert lo_s <= x_small.mean() <= hi_s
    assert lo_l <= x_large.mean() <= hi_l
    assert (hi_l - lo_l) < (hi_s - lo_s)

    p = paired_summary([80.0, 90.0, 70.0], [80.0, 90.0, 70.0])
    assert p["mean_diff"] == 0.0
    assert p["ci_lo"] == 0.0 and p["ci_hi"] == 0.0
    assert p["non_inferior_95"] is True


def test_make_report_from_fixture(tmp_path):
    """build_report renders REPORT.md from fixture JSON without needing ViZDoom."""
    import json
    from analysis.make_report import build_report

    (tmp_path / "chosen_config.json").write_text(json.dumps({
        "percentile": 99.99,
        "mem_init": 0.5,
        "T": 16,
        "thresholds": [0.4, 0.3, 0.2, 1.3],
        "s1_agreement": 0.92,
        "s1_regret": 0.001,
        "s2_snn_mean": 79.1,
        "s2_ann_mean": 79.3,
        "s2_kill_rate": 1.0,
    }), encoding="utf-8")

    (tmp_path / "results_summary.json").write_text(json.dumps({
        "n_episodes": 50,
        "paired": True,
        "baselines": {
            "random": {"mean_reward": -250.0, "std_reward": 50.0, "reward_ci_lo": -265.0, "reward_ci_hi": -235.0, "kill_rate": 0.1},
            "scripted": {"mean_reward": 75.0, "std_reward": 20.0, "reward_ci_lo": 69.0, "reward_ci_hi": 81.0, "kill_rate": 0.95},
            "ann": {"mean_reward": 80.0, "std_reward": 12.0, "reward_ci_lo": 76.0, "reward_ci_hi": 84.0, "kill_rate": 1.0},
        },
        "t_sweep": [{
            "T": 16, "mean_reward": 79.5, "std_reward": 12.5,
            "reward_ci_lo": 75.5, "reward_ci_hi": 83.5,
            "kill_rate": 1.0, "agreement": 0.96,
            "paired_vs_ann": {"mean_diff": -0.5, "ci_lo": -2.0, "ci_hi": 1.0, "ratio": 0.993, "non_inferior_95": True},
        }],
        "selected_T": 16,
        "paired_selected": {"mean_diff": -0.5, "ci_lo": -2.0, "ci_hi": 1.0, "ratio": 0.993, "non_inferior_95": True},
        "spike_summary": {
            "T": 16, "n_episodes": 20, "mean_spikes_per_decision": 12000.0,
            "layers": {"conv_0": {"firing_rate": 0.1, "temporal_sparsity": 0.9, "silent_neuron_frac": 0.2, "mean_spikes": 8000.0}},
            "ops": {"conv0_analog_macs": 100000.0, "ann_macs_spiking_layers": 500000.0, "snn_sops_spiking_layers": 150000.0, "sop_to_mac_ratio": 0.30},
        },
    }), encoding="utf-8")

    out = build_report(tmp_path, checkpoint="checkpoints/best.pt")
    text = out.read_text(encoding="utf-8")
    assert "ANN-to-SNN Conversion Report" in text
    assert "99.99" in text
    assert "PASS (95% non-inferior to ANN)" in text
    assert "12,000" in text
