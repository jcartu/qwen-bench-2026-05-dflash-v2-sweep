#!/usr/bin/env python3
"""Generate hero & section images for the 2026-05 BF16+DFlash v2-sweep study
via Gemini 3.1 nano-banana (`gemini-3.1-flash-image-preview`).

Same pattern as the qwen-bench hub generator. Run idempotently; skips files
that already exist with non-trivial size.
"""
import os
import pathlib
import sys

from google import genai

OUT = pathlib.Path(__file__).parent / "images"
OUT.mkdir(parents=True, exist_ok=True)

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]
)
MODEL = "gemini-3.1-flash-image-preview"

PROMPTS = {
    "study_hero": (
        "A wide cinematic 16:9 hero banner for a scientific paper on LLM inference "
        "parameter optimization. Centered: a glowing 3x3 grid heatmap visualization "
        "with one cell highlighted with a soft golden star, representing the 'winner' "
        "configuration. Below the grid, a sweeping line graph with a sharp downward "
        "cliff at the far right edge (representing a discovered performance cliff). "
        "Behind the visualizations, dual NVIDIA Blackwell-class data-center GPUs "
        "rendered in soft background bokeh, their cooling fins glowing faintly cyan. "
        "Floating microscopic particles suggest speculative-decoding draft tokens "
        "branching from a central node. Deep navy and charcoal background, electric "
        "cyan, amber, and a single magenta highlight on the cliff. Clean "
        "technical-research-paper aesthetic. No readable text, no logos, no "
        "watermarks. Style: precise modern scientific infographic meets cinematic "
        "3D render."
    ),
    "stage_a_section": (
        "An abstract illustration of a 3-by-3 grid of evaluation cells, each cell "
        "softly glowing with a slightly different shade of green, blue, and yellow, "
        "but all very close in brightness — visually communicating that the values "
        "are nearly the same. One single cell in the bottom-right corner is "
        "highlighted with a small golden star. Background is dark navy with subtle "
        "circuit-board traces. Minimal, clean, scientific-poster aesthetic. No "
        "readable text, no axis labels, no logos, no watermarks."
    ),
    "stage_b_section": (
        "An abstract horizontal line graph illustration with four distinct points "
        "from left to right. The first point is very low (catastrophe), the second "
        "point shoots up to peak height (winner), the third point dips slightly, "
        "and the fourth point falls off a steep cliff with a glowing magenta "
        "warning highlight at the cliff edge. The cliff is visually dramatic, "
        "almost like a physical drop. Dark navy background, neon cyan line, amber "
        "winner-glow on the peak. Minimal, clean, scientific-poster aesthetic. No "
        "readable text, no axis labels, no logos, no watermarks."
    ),
}

for name, prompt in PROMPTS.items():
    out_path = OUT / f"{name}.png"
    if out_path.exists() and out_path.stat().st_size > 1000:
        print(f"[skip] {out_path} exists ({out_path.stat().st_size} bytes)")
        continue
    print(f"[gen]  {name}: {prompt[:80]}...")
    try:
        resp = client.models.generate_content(model=MODEL, contents=prompt)
        wrote = False
        for part in resp.candidates[0].content.parts:
            if getattr(part, "inline_data", None) and part.inline_data.data:
                out_path.write_bytes(part.inline_data.data)
                print(f"[ok]   {out_path} ({out_path.stat().st_size} bytes)")
                wrote = True
                break
        if not wrote:
            print(f"[warn] no image data for {name}")
    except Exception as e:
        print(f"[err]  {name}: {e}", file=sys.stderr)
        sys.exit(1)
