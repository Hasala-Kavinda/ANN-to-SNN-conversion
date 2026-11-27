"""
Auto-generate REPORT.md from pipeline outputs in --results-dir.

Reads:
  - results_summary.json
  - chosen_config.json
  - sweep_results.csv (optional)
  - PNG charts in results_dir

Usage
-----
python analysis/make_report.py --results-dir results
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


def _fmt_ci(lo: float | None, hi: float | None, fmt: str = ".1f", scale: float = 1.0, suffix: str = "") -> str:
    if lo is None or hi is None:
        return "n/a"
    return f"[{lo * scale:{fmt}}{suffix}, {hi * scale:{fmt}}{suffix}]"


def build_report(
    results_dir: str | Path,
    out_path: str | Path | None = None,
    checkpoint: str | None = None,
) -> Path:
    rdir = Path(results_dir)
    out = Path(out_path) if out_path is not None else (rdir / "REPORT.md")

    summary_path = rdir / "results_summary.json"
    chosen_path = rdir / "chosen_config.json"
    csv_path = rdir / "sweep_results.csv"

    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    chosen = json.loads(chosen_path.read_text(encoding="utf-8")) if chosen_path.exists() else None

    lines: list[str] = []
    lines.append("# ANN-to-SNN Conversion Report\n")
    lines.append(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if checkpoint:
        lines.append(f"- **Source ANN checkpoint:** `{checkpoint}`")
    if summary:
        lines.append(
            f"- **Evaluation protocol:** {summary.get('n_episodes', '?')} episodes"
            + (" (paired per-episode seeds)" if summary.get("paired") else "")
        )
    lines.append("")

    # ── Chosen Configuration ──────────────────────────────────────────────────
    lines.append("## 1. Selected SNN Configuration\n")
    if chosen:
        lines.append(f"- **Percentile:** `{chosen.get('percentile')}`")
        lines.append(f"- **Membrane init ratio:** `{chosen.get('mem_init')}`")
        lines.append(f"- **Operating timesteps (T):** `{chosen.get('T')}`")
        thresholds = chosen.get("thresholds")
        if thresholds:
            t_str = ", ".join(f"{t:.4f}" for t in thresholds)
            lines.append(f"- **Layer thresholds:** `[{t_str}]`")
        if "s1_agreement" in chosen:
            lines.append(
                f"- **Stage 1 offline screen:** agreement `{chosen['s1_agreement']*100:.1f}%`, "
                f"mean Q-regret `{chosen.get('s1_regret', 0.0):.4f}`"
            )
        if "s2_snn_mean" in chosen and "s2_ann_mean" in chosen:
            lines.append(
                f"- **Stage 2 gameplay screen:** SNN `{chosen['s2_snn_mean']:.1f}` vs "
                f"ANN `{chosen['s2_ann_mean']:.1f}` (kill rate `{chosen.get('s2_kill_rate', 0.0)*100:.1f}%`)"
            )
        if "convert_cmd" in chosen:
            lines.append(f"\n```bash\n{chosen['convert_cmd']}\n```")
    else:
        lines.append("_`chosen_config.json` not found._")
    lines.append("")

    # ── Baselines & Paired Comparison ─────────────────────────────────────────
    lines.append("## 2. Baselines & Paired ANN-vs-SNN Comparison\n")
    if summary and "baselines" in summary:
        base = summary["baselines"]
        lines.append("| Agent | Mean Reward | 95% Bootstrap CI | Kill Rate | Notes |")
        lines.append("|---|---:|---|---:|---|")
        for key, label in [("random", "Random"), ("scripted", "Scripted"), ("ann", "ANN (DQN)")]:
            if key in base:
                b = base[key]
                ci = _fmt_ci(b.get("reward_ci_lo"), b.get("reward_ci_hi"))
                lines.append(
                    f"| {label} | {b['mean_reward']:.2f} | {ci} | "
                    f"{b['kill_rate']*100:.1f}% | ±{b['std_reward']:.1f} std |"
                )

        sel_T = summary.get("selected_T")
        sel_row = next((r for r in summary.get("t_sweep", []) if r.get("T") == sel_T), None)
        if sel_row:
            ci = _fmt_ci(sel_row.get("reward_ci_lo"), sel_row.get("reward_ci_hi"))
            lines.append(
                f"| **SNN (T={sel_T})** | **{sel_row['mean_reward']:.2f}** | {ci} | "
                f"{sel_row['kill_rate']*100:.1f}% | agreement {sel_row['agreement']*100:.1f}% |"
            )
        lines.append("")

        paired_sel = summary.get("paired_selected")
        if paired_sel:
            ratio_str = f"{paired_sel['ratio']*100:.1f}%" if paired_sel.get("ratio") is not None else "n/a"
            verdict = (
                "PASS (95% non-inferior to ANN)"
                if paired_sel.get("non_inferior_95")
                else "BELOW 95% non-inferiority threshold"
            )
            lines.append(
                f"**Paired difference (SNN − ANN at T={sel_T}):** "
                f"`{paired_sel['mean_diff']:+.2f}` "
                f"(95% CI `{_fmt_ci(paired_sel.get('ci_lo'), paired_sel.get('ci_hi'), fmt='+.2f')}`, "
                f"SNN/ANN ratio `{ratio_str}`) — **{verdict}**."
            )
    else:
        lines.append("_`results_summary.json` not found._")
    lines.append("")

    # ── T-Sweep Table ─────────────────────────────────────────────────────────
    lines.append("## 3. Timestep (T) Sweep\n")
    if summary and summary.get("t_sweep"):
        lines.append("| T | SNN Mean Reward | 95% CI | Paired Diff vs ANN | Diff 95% CI | Kill Rate | Agreement |")
        lines.append("|---:|---:|---|---:|---|---:|---:|")
        for r in summary["t_sweep"]:
            ci = _fmt_ci(r.get("reward_ci_lo"), r.get("reward_ci_hi"))
            p = r.get("paired_vs_ann") or {}
            diff_str = f"{p['mean_diff']:+.2f}" if "mean_diff" in p else "n/a"
            diff_ci = _fmt_ci(p.get("ci_lo"), p.get("ci_hi"), fmt="+.2f")
            lines.append(
                f"| {r['T']} | {r['mean_reward']:.2f} | {ci} | {diff_str} | {diff_ci} | "
                f"{r['kill_rate']*100:.1f}% | {r['agreement']*100:.1f}% |"
            )
    else:
        lines.append("_No T-sweep data available._")
    lines.append("")

    # ── Spike Activity & Compute ──────────────────────────────────────────────
    lines.append("## 4. Spiking Activity & Compute Accounting\n")
    if summary and summary.get("spike_summary"):
        sp = summary["spike_summary"]
        ops = sp.get("ops", {})
        lines.append(f"- **Evaluated at T:** `{sp.get('T')}` ({sp.get('n_episodes')} episodes)")
        lines.append(f"- **Mean spikes / decision:** `{sp.get('mean_spikes_per_decision', 0.0):,.0f}`")
        if ops:
            lines.append(f"- **Conv1 analog MACs (1×/decision):** `{ops.get('conv0_analog_macs', 0.0):,.0f}`")
            lines.append(f"- **ANN MACs (spike-driven layers):** `{ops.get('ann_macs_spiking_layers', 0.0):,.0f}`")
            lines.append(
                f"- **SNN SOPs (spike-driven layers):** `{ops.get('snn_sops_spiking_layers', 0.0):,.0f}` "
                f"(`{ops.get('sop_to_mac_ratio', 0.0):.2f}×` of ANN MACs)"
            )
        lines.append("")
        layers = sp.get("layers", {})
        if layers:
            lines.append("| Layer | Mean Firing Rate | Temporal Sparsity | Silent Neurons | Mean Spikes |")
            lines.append("|---|---:|---:|---:|---:|")
            for lname, lstats in layers.items():
                lines.append(
                    f"| `{lname}` | {lstats['firing_rate']:.4f} | "
                    f"{lstats['temporal_sparsity']*100:.1f}% | "
                    f"{lstats['silent_neuron_frac']*100:.1f}% | "
                    f"{lstats['mean_spikes']:,.0f} |"
                )
    else:
        lines.append("_No spike summary available._")
    lines.append("")

    # ── Stage 1 Sweep Summary ─────────────────────────────────────────────────
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            lines.append("## 5. Offline Conversion Sweep (Top 5 by Agreement / Regret)\n")
            top_rows = sorted(
                rows,
                key=lambda x: (-float(x.get("agreement", 0)), float(x.get("regret", 999))),
            )[:5]
            lines.append("| Percentile | Mem Init | T | Agreement | Q-Regret | MAE/Gap | Mean Clip % |")
            lines.append("|---:|---:|---:|---:|---:|---:|---:|")
            for r in top_rows:
                lines.append(
                    f"| {r['percentile']} | {r['mem_init']} | {r['T']} | "
                    f"{float(r['agreement'])*100:.1f}% | {float(r['regret']):.4f} | "
                    f"{float(r['q_mae_over_gap']):.2f} | {float(r['mean_clip_frac'])*100:.3f}% |"
                )
            lines.append("")

    # ── Charts ────────────────────────────────────────────────────────────────
    lines.append("## 6. Diagnostic & Result Charts\n")
    chart_specs = [
        ("sweep_heatmap.png", "Stage 1 Conversion Sweep Heatmap"),
        ("conversion_diagnostics.png", "Per-Layer ANN vs SNN Activation Diagnostics"),
        ("results_reward_vs_T.png", "Reward vs Timesteps T (95% CI)"),
        ("results_agreement_vs_T.png", "Action Agreement vs Timesteps T"),
        ("results_spike_rates.png", "Per-Layer Spike Firing Rates"),
    ]
    any_chart = False
    for fname, title in chart_specs:
        if (rdir / fname).exists():
            lines.append(f"### {title}\n![{title}]({fname})\n")
            any_chart = True
    if not any_chart:
        lines.append("_No chart PNGs found in results directory._\n")

    # ── Caveats ───────────────────────────────────────────────────────────────
    lines.append("## 7. Caveats & Scope\n")
    if summary and int(summary.get("n_episodes", 0)) < 30:
        lines.append(
            f"- **Low-sample run:** Evaluated on {summary.get('n_episodes')} episodes (`--quick` or smoke test); "
            "confidence intervals are wide and results are indicative only."
        )
    lines.append(
        "- **Scope:** Software-level ANN-to-SNN rate/IF conversion and analytical SOP accounting only. "
        "Fixed-point weight/membrane quantization for FPGA deployment is out of scope."
    )
    lines.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved: {out}")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate REPORT.md from pipeline results")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--out",         type=str, default=None)
    p.add_argument("--checkpoint",  type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_report(args.results_dir, out_path=args.out, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
