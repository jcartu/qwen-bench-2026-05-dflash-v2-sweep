#!/usr/bin/env python3
"""Pick Stage B winner by aggregate decode tok/s (same metric as Stage A)."""
import json
import re
import statistics
import sys
from pathlib import Path

STUDY_ROOT = Path("/tmp/qwen-bench-2026-05-dflash-v2-sweep")
STAGE_DIR = STUDY_ROOT / "configs/stage-b"
LABEL_RE = re.compile(r"b(\d+)_c(\d+)_n(\d+)")


def extract_cells(p: Path) -> list[float]:
    data = json.loads(p.read_text())
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
    ranking: list[tuple[str, float, int, int, int]] = []
    for cfg_dir in sorted(STAGE_DIR.iterdir()):
        if not cfg_dir.is_dir() or cfg_dir.name.startswith("_"):
            continue
        tp = cfg_dir / "throughput.json"
        if not tp.exists():
            continue
        m = LABEL_RE.match(cfg_dir.name)
        if not m:
            continue
        b, c, n = (int(x) for x in m.groups())
        try:
            cells = extract_cells(tp)
        except Exception:
            continue
        if not cells:
            continue
        ranking.append((cfg_dir.name, statistics.mean(cells), b, c, n))

    if not ranking:
        print("ERROR: no successful Stage B configs", file=sys.stderr)
        return 1

    ranking.sort(key=lambda r: r[1], reverse=True)

    csv = STAGE_DIR / "_ranking.csv"
    with csv.open("w") as fh:
        fh.write("rank,label,batched,capture,num_spec,mean_decode_tps\n")
        for i, (label, mean, b, c, n) in enumerate(ranking, 1):
            fh.write(f"{i},{label},{b},{c},{n},{mean:.2f}\n")

    win_label, win_mean, win_b, win_c, win_n = ranking[0]
    (STAGE_DIR / "_winner.txt").write_text(
        f"WIN_BATCHED={win_b}\nWIN_CAPTURE={win_c}\nWIN_NUM_SPEC={win_n}\n"
        f"WIN_LABEL={win_label}\nWIN_MEAN_TPS={win_mean:.2f}\n"
    )

    print(f"\n[STAGE B RANKING]")
    for i, (label, mean, b, c, n) in enumerate(ranking, 1):
        marker = "★" if i == 1 else " "
        print(f"  {marker} #{i} {label:30s} mean={mean:7.1f} tok/s")
    print(f"\nWinner: {win_label} → batched={win_b} capture={win_c} num_spec={win_n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
