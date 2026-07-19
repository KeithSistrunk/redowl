"""Generates assets/three_arm_results.png -- the bar chart for the README's
"Results -- Three-Arm Constraint Experiment" section.

Not part of the shipped CLI. Built following the project's dataviz skill:
form = horizontal bar (magnitude across 3 nominal categories), color = fixed
categorical order (palette.md slots 1-3, validated with
scripts/validate_palette.js before use here), thin marks with rounded data
ends, hairline solid gridlines, direct value labels (required by the
palette's own "relief rule" for the aqua/yellow slots), no legend (single
series -- the arm names on the y-axis already say what's plotted). The
README table is this chart's table-view twin.

Static PNG, not an interactive HTML chart: this embeds in a GitHub README,
which renders plain images, not JS/CSS. Colors are picked for the light
chart surface only (GitHub's default), noted as a known limitation rather
than silently pretending dark-mode parity that a flat PNG can't deliver.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "assets" / "three_arm_results.png"

# palette.md categorical slots 1-3, validated light-mode PASS via
# scripts/validate_palette.js "#2a78d6,#1baf7a,#eda100" --mode light
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

ARMS = [
    {"label": "C — Free-gen", "rate": 11, "lo": 0, "hi": 18, "color": "#eda100", "n": "6/53"},
    {"label": "B — Variant (GPT)", "rate": 31, "lo": 18, "hi": 45, "color": "#1baf7a", "n": "14/45"},
    {"label": "A — Verbatim pool", "rate": 45, "lo": 36, "hi": 64, "color": "#2a78d6", "n": "25/55"},
]

BAR_HEIGHT = 0.42

fig, ax = plt.subplots(figsize=(9, 4.3), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

y_positions = list(range(len(ARMS)))

# Gridlines first (recessive, behind marks), solid hairlines only.
for x in (0, 20, 40, 60):
    ax.axvline(x, color=GRIDLINE, linewidth=1, zorder=1)

for y, arm in zip(y_positions, ARMS):
    # Bar, square ends -- a rounded data-end via FancyBboxPatch distorted
    # into a curved artifact here because x (0-78, percent) and y
    # (0-2, category index) are wildly different-scale axes, so a
    # data-coordinate rounding_size isn't uniform in display space. A plain
    # rect avoids that; not worth fighting the transform for a 3-bar chart.
    ax.barh(y, arm["rate"], height=BAR_HEIGHT, color=arm["color"], zorder=3, linewidth=0)

    # Per-run range: thin secondary-ink whisker with end caps, drawn through
    # the bar -- a secondary annotation, not a second series (no legend
    # entry), explained in the caption below the chart.
    ax.plot([arm["lo"], arm["hi"]], [y, y], color=INK_SECONDARY, linewidth=1.4, zorder=4, solid_capstyle="butt")
    for x in (arm["lo"], arm["hi"]):
        ax.plot([x, x], [y - 0.09, y + 0.09], color=INK_SECONDARY, linewidth=1.4, zorder=4)

    # Direct label, positioned clear of BOTH the bar tip and the range
    # whisker (whichever extends further) so it never collides with the
    # whisker's end-cap -- the first draft put this at the bar tip alone and
    # the whisker crossed straight through the text.
    label_x = max(arm["rate"], arm["hi"]) + 2.4
    ax.text(
        label_x, y, f"{arm['rate']}% ({arm['n']})",
        va="center", ha="left", fontsize=11, color=INK_PRIMARY, fontweight="medium",
    )

ax.set_yticks(y_positions)
ax.set_yticklabels([a["label"] for a in ARMS], fontsize=11.5, color=INK_PRIMARY)
ax.tick_params(axis="y", length=0)

ax.set_xlim(0, 92)
ax.set_xticks([0, 20, 40, 60])
ax.set_xticklabels(["0%", "20%", "40%", "60%"], fontsize=9.5, color=INK_MUTED)
ax.tick_params(axis="x", length=0, pad=6)

ax.set_ylim(-0.7, len(ARMS) - 0.3)

# Baseline (x=0), solid hairline in baseline ink -- not a heavier axis rule.
ax.axvline(0, color=BASELINE, linewidth=1, zorder=2)

for spine in ax.spines.values():
    spine.set_visible(False)

fig.suptitle(
    "Attack success rate by constraint level",
    x=0.02, y=0.98, ha="left", fontsize=14.5, color=INK_PRIMARY, fontweight="semibold",
)
ax.set_title(
    "prompt_injection objective · llama3.2 target · 5 runs per arm",
    loc="left", fontsize=10.5, color=INK_SECONDARY, pad=14,
)

fig.text(
    0.02, 0.01,
    "Bar = pooled success rate across all runs. Thin line = per-run range.  "
    "Runs B-1 and C-4 partial (JSON-parse failures).",
    fontsize=8.5, color=INK_MUTED, ha="left",
)

fig.tight_layout(rect=(0, 0.05, 1, 0.93))
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, facecolor=SURFACE)
print(f"Wrote {OUT_PATH}")
