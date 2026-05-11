#!/usr/bin/env python3
"""Pick Stage A winner by aggregate decode tok/s averaged over 15 cells.

Reads every configs/stage-a/<LABEL>/throughput.json, computes the mean of
all (concurrency, context) cells, ranks, writes:
  - configs/stage-a/_winner.txt   (shell-sourceable: WIN_BATCHED=..., WIN_CAPTURE=...)
  - configs/stage-a/_ranking.csv  (full ranking)
"""
import json
import re
import statistics
import sys
from pathlib import Path

STUDY_ROOT = Path("/tmp/qwen-bench-2026-05-dflash-v2-sweep")
STAGE_DIR = STUDY_ROOT / "configs/stage-a"

LABEL_RE = re.compile(r"b(\d+)_c(\d+)_n(\d+)")


def extract_cells(throughput_json_path: Path) -> list[float]:
    """Return decode aggregate tok/s for every (conc, ctx) cell in the file.

    llm_decode_bench.py output schema (verified 2026-05-11 smoke test):
      data['results'] is a list of cells; each has 'aggregate_tps' as the
      headline decode throughput. We skip prefill-mode cells if any leak in.
    """
    data = json.loads(throughput_json_path.read_text())
    cells: list[float] = []
    for entry in data.get("results", []):
        if entry.get("benchmark_mode") == "prefill":
            continue
        v = entry.get("aggregate_tps")
        if v is None or v <= 0:
            continue
        cells.append(float(v))
    return cells


def main() -> int:
    ranking: list[tuple[str, float, int, int, list[float]]] = []
    for cfg_dir in sorted(STAGE_DIR.iterdir()):
        if not cfg_dir.is_dir() or cfg_dir.name.startswith("_"):
            continue
        tp = cfg_dir / "throughput.json"
        if not tp.exists():
            print(f"  [skip] {cfg_dir.name}: no throughput.json", file=sys.stderr)
            continue
        m = LABEL_RE.match(cfg_dir.name)
        if not m:
            continue
        batched, capture, _ = (int(x) for x in m.groups())
        try:
            cells = extract_cells(tp)
        except Exception as exc:
            print(f"  [skip] {cfg_dir.name}: parse error {exc}", file=sys.stderr)
            continue
        if not cells:
            print(f"  [skip] {cfg_dir.name}: 0 cells parsed", file=sys.stderr)
            continue
        mean_tps = statistics.mean(cells)
        ranking.append((cfg_dir.name, mean_tps, batched, capture, cells))

    if not ranking:
        print("ERROR: no successful Stage A configs found", file=sys.stderr)
        return 1

    ranking.sort(key=lambda r: r[1], reverse=True)

    csv_path = STAGE_DIR / "_ranking.csv"
    with csv_path.open("w") as fh:
        fh.write("rank,label,batched,capture,mean_decode_tps,n_cells\n")
        for i, (label, mean_tps, batched, capture, cells) in enumerate(ranking, 1):
            fh.write(f"{i},{label},{batched},{capture},{mean_tps:.2f},{len(cells)}\n")

    win_label, win_mean, win_b, win_c, _ = ranking[0]
    win_txt = STAGE_DIR / "_winner.txt"
    win_txt.write_text(f"WIN_BATCHED={win_b}\nWIN_CAPTURE={win_c}\nWIN_LABEL={win_label}\nWIN_MEAN_TPS={win_mean:.2f}\n")

    print(f"\n[STAGE A RANKING]")
    for i, (label, mean_tps, batched, capture, cells) in enumerate(ranking, 1):
        marker = "★" if i == 1 else " "
        print(f"  {marker} #{i} {label:30s} mean={mean_tps:7.1f} tok/s (n={len(cells)} cells)")
    print(f"\nWinner: {win_label} → batched={win_b} capture={win_c}")
    print(f"Wrote: {win_txt}")
    print(f"Wrote: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
