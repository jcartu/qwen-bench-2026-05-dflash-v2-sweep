#!/usr/bin/env python3
"""Generate the three headline plots for the study:
  1. docs/images/stage_a_heatmap.png — Stage A 3×3 buffer/graph heatmap
  2. docs/images/stage_b_curve.png   — Stage B num_speculative_tokens curve
  3. docs/images/per_cell_heatmap.png — Stage B winner per-cell decode heatmap
"""
import json
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

ROOT = Path("/tmp/qwen-bench-2026-05-dflash-v2-sweep")
OUT = ROOT / "docs/images"
OUT.mkdir(parents=True, exist_ok=True)

LABEL_RE = re.compile(r"b(\d+)_c(\d+)_n(\d+)")


def load_cells(p: Path) -> list[float]:
    d = json.loads(p.read_text())
    return [
        r["aggregate_tps"]
        for r in d["results"]
        if r.get("benchmark_mode") != "prefill" and r.get("aggregate_tps", 0) > 0
    ]


def load_full(p: Path) -> list[dict]:
    return [
        r for r in json.loads(p.read_text())["results"]
        if r.get("benchmark_mode") != "prefill"
    ]


# ── PLOT 1: Stage A 3×3 heatmap ──────────────────────────────────────────────
batched_vals = [8192, 16384, 32768]
capture_vals = [64, 128, 256]
grid = np.zeros((3, 3))
for i, b in enumerate(batched_vals):
    for j, c in enumerate(capture_vals):
        d = ROOT / f"configs/stage-a/b{b}_c{c}_n8"
        cells = load_cells(d / "throughput.json")
        grid[i, j] = statistics.mean(cells)

fig, ax = plt.subplots(figsize=(7, 5.5))
im = ax.imshow(grid, cmap="RdYlGn", aspect="auto",
               vmin=grid.min() - 1, vmax=grid.max() + 1)
ax.set_xticks(range(3), [str(c) for c in capture_vals])
ax.set_yticks(range(3), [f"{b:,}" for b in batched_vals])
ax.set_xlabel("--max-cudagraph-capture-size", fontsize=11)
ax.set_ylabel("--max-num-batched-tokens", fontsize=11)
ax.set_title(
    "Stage A — Buffer/Graph Sweep (num_spec=8)\nMean aggregate decode tok/s across 15 cells",
    fontsize=12,
)
for i in range(3):
    for j in range(3):
        v = grid[i, j]
        is_winner = v == grid.max()
        ax.text(
            j, i, f"{v:.1f}{'  ★' if is_winner else ''}",
            ha="center", va="center",
            color="black", fontsize=11,
            fontweight="bold" if is_winner else "normal",
        )
cb = plt.colorbar(im, ax=ax)
cb.set_label("tok/s", fontsize=10)
ax.text(
    0.5, -0.18,
    f"Range: {grid.min():.1f} – {grid.max():.1f} tok/s (Δ={100*(grid.max()-grid.min())/grid.min():.1f}%)\n"
    "Conclusion: buffer/graph parameters do not meaningfully move decode throughput for BF16+DFlash.",
    transform=ax.transAxes, ha="center", fontsize=9, style="italic",
)
plt.tight_layout()
plt.savefig(OUT / "stage_a_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ {OUT/'stage_a_heatmap.png'}")


# ── PLOT 2: Stage B num_spec curve ───────────────────────────────────────────
num_spec_vals = [4, 8, 15, 16]
stage_b = {}
for n in num_spec_vals:
    d = ROOT / f"configs/stage-b/b32768_c256_n{n}"
    rows = load_full(d / "throughput.json")
    cells = [r["aggregate_tps"] for r in rows if r.get("aggregate_tps", 0) > 0]
    rates = [r.get("server_spec_accept_rate", 0) for r in rows]
    stage_b[n] = {
        "mean_tps": statistics.mean(cells),
        "min_tps": min(cells),
        "max_tps": max(cells),
        "accept_rate": statistics.mean(rates),
    }

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left: tok/s
xs = num_spec_vals
ys = [stage_b[n]["mean_tps"] for n in xs]
mns = [stage_b[n]["min_tps"] for n in xs]
mxs = [stage_b[n]["max_tps"] for n in xs]
ax1.plot(xs, ys, "o-", color="#1f77b4", linewidth=2.5, markersize=9, label="mean across 15 cells")
ax1.fill_between(xs, mns, mxs, color="#1f77b4", alpha=0.15, label="min-max range")
ax1.set_xlabel("--speculative-config.num_speculative_tokens", fontsize=11)
ax1.set_ylabel("Aggregate decode tok/s", fontsize=11)
ax1.set_title("Stage B — Decode throughput vs draft length", fontsize=12)
ax1.set_xticks(xs)
ax1.grid(True, alpha=0.3)
# Annotate winner
win_n = max(stage_b, key=lambda k: stage_b[k]["mean_tps"])
win_y = stage_b[win_n]["mean_tps"]
ax1.annotate(
    f"★ winner\nn={win_n}, {win_y:.1f} tok/s",
    xy=(win_n, win_y), xytext=(win_n + 1, win_y - 35),
    fontsize=10, ha="left",
    arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
)
# Annotate cliff
ax1.annotate(
    "−42.6% cliff",
    xy=(16, stage_b[16]["mean_tps"]), xytext=(13.5, 85),
    fontsize=10, ha="center", color="red", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
)
ax1.legend(loc="upper right", fontsize=9)

# Right: spec accept rate
ar_ys = [stage_b[n]["accept_rate"] * 100 for n in xs]
ax2.bar(xs, ar_ys, color=["#d62728", "#2ca02c", "#ff7f0e", "#d62728"], alpha=0.8)
for x, y in zip(xs, ar_ys):
    ax2.text(x, y + 0.5, f"{y:.1f}%", ha="center", fontsize=10)
ax2.set_xlabel("--speculative-config.num_speculative_tokens", fontsize=11)
ax2.set_ylabel("Server spec acceptance rate (%)", fontsize=11)
ax2.set_title("Drafter acceptance rate by draft length", fontsize=12)
ax2.set_xticks(xs)
ax2.grid(True, alpha=0.3, axis="y")
ax2.set_ylim(0, max(ar_ys) * 1.2)

fig.suptitle(
    "Stage B — Speculative-tokens sweep (batched=32768, capture=256)",
    fontsize=13, y=1.02,
)
plt.tight_layout()
plt.savefig(OUT / "stage_b_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ {OUT/'stage_b_curve.png'}")


# ── PLOT 3: Per-cell heatmap for the overall winner ─────────────────────────
win_dir = ROOT / "configs/stage-b/b32768_c256_n8"
rows = load_full(win_dir / "throughput.json")
ctxs = sorted({r["context_tokens"] for r in rows})
concs = sorted({r["concurrency"] for r in rows})
cell = np.zeros((len(ctxs), len(concs)))
for r in rows:
    if r.get("aggregate_tps", 0) > 0:
        ci = ctxs.index(r["context_tokens"])
        cj = concs.index(r["concurrency"])
        cell[ci, cj] = r["aggregate_tps"]

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cell, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(concs)), [str(c) for c in concs])
ax.set_yticks(range(len(ctxs)), [f"{c//1024}k" if c >= 1024 else str(c) for c in ctxs])
ax.set_xlabel("concurrency", fontsize=11)
ax.set_ylabel("context tokens", fontsize=11)
ax.set_title(
    "Winner config (b32768_c256_n8) — aggregate decode tok/s by cell",
    fontsize=12,
)
for i in range(len(ctxs)):
    for j in range(len(concs)):
        v = cell[i, j]
        ax.text(
            j, i, f"{v:.0f}",
            ha="center", va="center",
            color="white" if v < cell.max() * 0.6 else "black", fontsize=10,
        )
cb = plt.colorbar(im, ax=ax)
cb.set_label("tok/s", fontsize=10)
plt.tight_layout()
plt.savefig(OUT / "per_cell_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ {OUT/'per_cell_heatmap.png'}")
print(f"\nAll plots written to {OUT}")
