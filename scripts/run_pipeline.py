"""
One-command end-to-end ANN-to-SNN conversion, evaluation, video, and report pipeline.

Stages
------
  1. calibrate  — collect per-layer ReLU stats + offline evaluation frames
  2. sweep      — Stage 1 offline screen + Stage 2 paired gameplay confirmation
  3. convert    — build & verify SNN using chosen_config.json
  4. evaluate   — paired ANN & SNN evaluation with 95% bootstrap CIs
  5. plots      — result charts (reward/agreement/spikes + conversion diagnostics)
  6. videos     — ANN, SNN, and side-by-side --compare MP4 videos
  7. report     — auto-generate REPORT.md

Usage
-----
python scripts/run_pipeline.py --checkpoint checkpoints/best.pt
python scripts/run_pipeline.py --checkpoint checkpoints_local/best.pt --quick
python scripts/run_pipeline.py --checkpoint checkpoints_local/best.pt --from-stage report
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STAGES = ["calibrate", "sweep", "convert", "evaluate", "plots", "videos", "report"]


def _resolve_stage_index(stage: str) -> int:
    s = stage.strip().lower()
    if s.isdigit():
        idx = int(s) - 1
        if 0 <= idx < len(STAGES):
            return idx
    if s in STAGES:
        return STAGES.index(s)
    raise ValueError(f"Unknown stage '{stage}'. Valid: 1..{len(STAGES)} or {STAGES}")


def _run(cmd: list[str], stage_name: str) -> float:
    print(f"\n{'='*70}")
    print(f"[Pipeline] Stage: {stage_name}")
    print(f"[Pipeline] Cmd:   {' '.join(cmd)}")
    print(f"{'='*70}\n")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    dt = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"Stage '{stage_name}' failed with exit code {proc.returncode}:\n  {' '.join(cmd)}"
        )
    print(f"[Pipeline] Stage '{stage_name}' completed in {dt:.1f}s")
    return dt


def _require_files(paths: list[Path], stage_name: str) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Cannot start at stage '{stage_name}'; missing prerequisite file(s): {missing}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end ANN-to-SNN pipeline")
    p.add_argument("--checkpoint",  type=str, default="checkpoints/best.pt")
    p.add_argument("--quick",       action="store_true",
                   help="Fast smoke-test budget (~minutes)")
    p.add_argument("--from-stage",  type=str, default="1",
                   help="Resume from stage number (1-7) or name")
    p.add_argument("--results-dir", type=str, default=None)
    p.add_argument("--video-dir",   type=str, default=None)
    p.add_argument("--snn-out",     type=str, default=None)
    p.add_argument("--seed",        type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    py = sys.executable

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = PROJECT_ROOT / ckpt
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    ckpt_dir = ckpt.parent
    is_local = "checkpoints_local" in ckpt.parts

    results_dir = Path(args.results_dir) if args.results_dir else PROJECT_ROOT / ("results_local" if is_local else "results")
    video_dir   = Path(args.video_dir)   if args.video_dir   else PROJECT_ROOT / ("videos_local"  if is_local else "videos")
    snn_out     = Path(args.snn_out)     if args.snn_out     else ckpt_dir / "best_snn.pt"
    stats_path  = ckpt_dir / "cal_stats.pkl"
    frames_path = ckpt_dir / "cal_frames.npz"
    chosen_path = results_dir / "chosen_config.json"

    results_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    start_idx = _resolve_stage_index(args.from_stage)

    # Budget settings
    if args.quick:
        cal_eps      = 20
        save_frames  = 1000
        pcts         = ["99.9", "99.99"]
        mem_inits    = ["0.0", "0.5"]
        T_vals       = ["8", "16", "32"]
        top_k        = 2
        sweep_eps    = 5
        verify_eps   = 5
        eval_eps     = 10
        agree_eps    = 10
        video_eps    = 1
    else:
        cal_eps      = 500
        save_frames  = 4000
        pcts         = ["99.0", "99.5", "99.9", "99.99", "100.0"]
        mem_inits    = ["0.0", "0.5"]
        T_vals       = ["8", "16", "32", "64", "128"]
        top_k        = 4
        sweep_eps    = 50
        verify_eps   = 50
        eval_eps     = 50
        agree_eps    = 20
        video_eps    = 3

    # Prerequisite checks when resuming mid-pipeline
    prereqs = {
        "sweep":    [stats_path, frames_path],
        "convert":  [stats_path, chosen_path],
        "evaluate": [snn_out, chosen_path],
        "plots":    [snn_out, chosen_path, frames_path],
        "videos":   [snn_out, chosen_path],
        "report":   [results_dir / "results_summary.json"],
    }
    start_stage = STAGES[start_idx]
    if start_stage in prereqs:
        _require_files(prereqs[start_stage], start_stage)

    state_path = results_dir / ".pipeline_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"stages": {}}
    timings: dict[str, float] = {}

    def _record_state(name: str, dt: float) -> None:
        timings[name] = round(dt, 2)
        state["stages"][name] = {"duration_s": round(dt, 2), "quick": args.quick}
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    # 1. Calibrate
    if start_idx <= 0:
        dt = _run([
            py, "conversion/calibrate.py",
            "--checkpoint", str(ckpt),
            "--episodes", str(cal_eps),
            "--save-frames", str(save_frames),
            "--out", str(stats_path),
            "--frames-out", str(frames_path),
        ], "calibrate")
        _record_state("calibrate", dt)

    # 2. Sweep
    if start_idx <= 1:
        dt = _run([
            py, "conversion/sweep.py",
            "--checkpoint", str(ckpt),
            "--stats", str(stats_path),
            "--frames", str(frames_path),
            "--percentiles", *pcts,
            "--mem-inits", *mem_inits,
            "--T-values", *T_vals,
            "--top-k", str(top_k),
            "--episodes", str(sweep_eps),
            "--out-dir", str(results_dir),
        ], "sweep")
        _record_state("sweep", dt)

    # Load chosen_config.json for stages 3..6
    _require_files([chosen_path], "convert")
    chosen = json.loads(chosen_path.read_text(encoding="utf-8"))
    chosen_pct = str(chosen["percentile"])
    chosen_mem = str(chosen["mem_init"])
    chosen_T   = str(chosen["T"])

    # 3. Convert
    if start_idx <= 2:
        dt = _run([
            py, "conversion/convert.py",
            "--checkpoint", str(ckpt),
            "--stats", str(stats_path),
            "--out", str(snn_out),
            "--no-calibrate",
            "--percentile", chosen_pct,
            "--mem-init", chosen_mem,
            "--verify-T", chosen_T,
            "--verify-eps", str(verify_eps),
        ], "convert")
        _record_state("convert", dt)

    # 4. Evaluate (paired ANN & SNN)
    if start_idx <= 3:
        t_eval = time.time()
        _run([
            py, "evaluation/eval_agent.py",
            "--mode", "ann",
            "--checkpoint", str(ckpt),
            "--episodes", str(eval_eps),
            "--paired",
            "--seed", str(args.seed),
            "--json-out", str(results_dir / "eval_ann.json"),
        ], "evaluate (ANN)")
        _run([
            py, "evaluation/eval_agent.py",
            "--mode", "snn",
            "--checkpoint", str(snn_out),
            "--config", str(chosen_path),
            "--episodes", str(eval_eps),
            "--paired",
            "--seed", str(args.seed),
            "--json-out", str(results_dir / "eval_snn.json"),
        ], "evaluate (SNN)")
        _record_state("evaluate", time.time() - t_eval)

    # 5. Plots + diagnostics
    if start_idx <= 4:
        t_plots = time.time()
        _run([
            py, "analysis/plot_results.py",
            "--ann-checkpoint", str(ckpt),
            "--snn-checkpoint", str(snn_out),
            "--config", str(chosen_path),
            "--T-values", *T_vals,
            "--episodes", str(eval_eps),
            "--agreement-eps", str(agree_eps),
            "--out-dir", str(results_dir),
            "--seed", str(args.seed),
        ], "plots (results)")
        _run([
            py, "analysis/conversion_diagnostics.py",
            "--ann-checkpoint", str(ckpt),
            "--snn-checkpoint", str(snn_out),
            "--frames", str(frames_path),
            "--T", chosen_T,
            "--out", str(results_dir / "conversion_diagnostics.png"),
        ], "plots (diagnostics)")
        _record_state("plots", time.time() - t_plots)

    # 6. Videos
    if start_idx <= 5:
        t_vid = time.time()
        _run([
            py, "evaluation/record_video.py",
            "--mode", "ann",
            "--checkpoint", str(ckpt),
            "--episodes", str(video_eps),
            "--out-dir", str(video_dir),
            "--seed", str(args.seed),
        ], "videos (ANN)")
        _run([
            py, "evaluation/record_video.py",
            "--mode", "snn",
            "--checkpoint", str(snn_out),
            "--T", chosen_T,
            "--episodes", str(video_eps),
            "--out-dir", str(video_dir),
            "--seed", str(args.seed),
        ], "videos (SNN)")
        _run([
            py, "evaluation/record_video.py",
            "--compare",
            "--checkpoint", str(ckpt),
            "--snn-checkpoint", str(snn_out),
            "--T", chosen_T,
            "--episodes", str(video_eps),
            "--out-dir", str(video_dir),
            "--seed", str(args.seed),
        ], "videos (compare)")
        _record_state("videos", time.time() - t_vid)

    # 7. Report
    if start_idx <= 6:
        dt = _run([
            py, "analysis/make_report.py",
            "--results-dir", str(results_dir),
            "--checkpoint", str(ckpt),
        ], "report")
        _record_state("report", dt)

    # Summary
    print(f"\n{'='*70}")
    print("Pipeline complete ✓")
    for k, v in timings.items():
        print(f"  {k:<12s}: {v:6.1f}s")

    expected = [
        results_dir / "sweep_results.csv",
        results_dir / "sweep_heatmap.png",
        results_dir / "chosen_config.json",
        results_dir / "results_reward_vs_T.png",
        results_dir / "results_agreement_vs_T.png",
        results_dir / "results_spike_rates.png",
        results_dir / "conversion_diagnostics.png",
        results_dir / "results_summary.json",
        results_dir / "REPORT.md",
        video_dir / "ann_episode.mp4",
        video_dir / f"snn_T{chosen_T}_episode.mp4",
        video_dir / f"compare_T{chosen_T}.mp4",
    ]
    print("\nArtefact checklist:")
    for p in expected:
        ok = "✓" if p.exists() and p.stat().st_size > 0 else "✗"
        print(f"  [{ok}] {p}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
