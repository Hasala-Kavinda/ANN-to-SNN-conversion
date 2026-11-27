# ANN-to-SNN Conversion for ViZDoom

End-to-end pipeline for training a DQN agent on ViZDoom (`basic`), converting it to an Integrate-and-Fire Spiking Neural Network (SNN) via threshold normalization, sweeping conversion parameters, and generating paired evaluation metrics, comparison videos, and a markdown report.

## Setup

### Windows (PowerShell, recommended)
```powershell
.\setup_win.ps1
.\venv_win\Scripts\Activate.ps1
```

### Linux / WSL
```bash
bash setup.sh
source venv/bin/activate
```

Core platform-neutral dependencies live in `requirements.txt` (PyTorch CUDA wheels are installed by the setup scripts).

## One-Command Pipeline

Once a trained ANN checkpoint (`checkpoints/best.pt`) is available, run the full pipeline:

```powershell
# Full run (500 calibration episodes, full T-sweep, 50 paired evaluation episodes)
python scripts/run_pipeline.py --checkpoint checkpoints/best.pt

# Quick smoke run (~minutes, for local verification)
python scripts/run_pipeline.py --checkpoint checkpoints_local/best.pt --quick

# Resume from a specific stage (calibrate, sweep, convert, evaluate, plots, videos, report)
python scripts/run_pipeline.py --checkpoint checkpoints/best.pt --from-stage evaluate
```

Checkpoints under `checkpoints_local/` automatically write outputs to `results_local/` and `videos_local/`, keeping `results/` and `videos/` clean for the final `checkpoints/best.pt` run.

## Pipeline Stages (Manual Commands)

1. **Calibrate** activations and save offline evaluation frames:
   ```bash
   python conversion/calibrate.py --checkpoint checkpoints/best.pt
   ```
2. **Two-stage sweep** (Stage 1 offline screen on saved frames, Stage 2 paired gameplay confirmation):
   ```bash
   python conversion/sweep.py --checkpoint checkpoints/best.pt --stats calibration_stats.pkl --frames calibration_frames.npz
   ```
3. **Convert** ANN to SNN using `chosen_config.json` settings:
   ```bash
   python conversion/convert.py --checkpoint checkpoints/best.pt --no-calibrate --percentile 99.99 --mem-init 0.5 --verify-T 16
   ```
4. **Evaluate** with paired per-episode seeds and 95% bootstrap confidence intervals:
   ```bash
   python evaluation/eval_agent.py --mode ann --checkpoint checkpoints/best.pt --episodes 50 --paired
   python evaluation/eval_agent.py --mode snn --checkpoint checkpoints/best_snn.pt --config results/chosen_config.json --episodes 50 --paired
   ```
5. **Plots & Diagnostics**:
   ```bash
   python analysis/plot_results.py --ann-checkpoint checkpoints/best.pt --snn-checkpoint checkpoints/best_snn.pt --config results/chosen_config.json
   python analysis/conversion_diagnostics.py --ann-checkpoint checkpoints/best.pt --snn-checkpoint checkpoints/best_snn.pt --frames calibration_frames.npz --T 16
   ```
6. **Videos** (single-agent or side-by-side `--compare`):
   ```bash
   python evaluation/record_video.py --compare --checkpoint checkpoints/best.pt --snn-checkpoint checkpoints/best_snn.pt --T 16 --episodes 3
   ```
7. **Report**:
   ```bash
   python analysis/make_report.py --results-dir results --checkpoint checkpoints/best.pt
   ```

## Reading the Results

- `results/chosen_config.json`: Selected `percentile`, `mem_init`, `T`, and per-layer thresholds.
- `results/sweep_heatmap.png`: Stage 1 offline action agreement and Q-regret across `(percentile, T)`.
- `results/conversion_diagnostics.png`: Per-layer ANN activation vs. SNN firing rate correlation, clip fraction, and dead neurons.
- `results/results_summary.json` & `results/REPORT.md`: Paired ANN-vs-SNN reward comparison with 95% bootstrap CIs, T-sweep table, and analytical SOPs vs. MACs.
- `videos/compare_T<T>.mp4`: Side-by-side gameplay on identical per-episode seeds with live Q-values, agreement/divergence flag, and spikes per decision.

## Running Tests

```powershell
# Fast unit tests (excludes 100-episode ViZDoom rollout)
pytest tests/ -m "not slow" -v

# Full suite including 100-episode rollout
pytest tests/ -v
```

The fast suite covers environment wrappers, ANN/SNN shapes, `uint8` replay buffer round-trip, `conv1` constant-current caching equivalence, per-layer threshold normalization, Stage 1 offline sweep, bootstrap/paired statistics, and report generation.

## Scope Note

This repository covers **software-level ANN-to-SNN rate/IF conversion** (constant-current direct input, reset-by-subtraction IF neurons, and analytical Synaptic Operations vs. MACs accounting). Fixed-point weight/membrane quantization for FPGA hardware deployment is out of scope.
