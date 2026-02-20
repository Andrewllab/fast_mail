#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# -------------------------
# 1. DATA (same as before)
# -------------------------

group_labels = ["Group 1", "Group 2"]

pointpatch_no = [0.80, 0.78]
pointpatch_ff = [0.85, 0.83]

dp3_no = [0.75, 0.74]
dp3_ff = [0.82, 0.80]

pcm_no = [0.70, 0.69]
pcm_ff = [0.79, 0.77]

pointmap_no = [0.77, 0.76]

data_arrays = [
    pointpatch_no,
    pointpatch_ff,
    dp3_no,
    dp3_ff,
    pcm_no,
    pcm_ff,
    pointmap_no,
]

labels_per_bar = [
    "PP noFF",
    "PP FF",
    "DP3 noFF",
    "DP3 FF",
    "PCM noFF",
    "PCM FF",
    "PointMap",
]

# -------------------------
# 2. LAYOUT / SPACING (same logic)
# -------------------------

n_groups = len(group_labels)

group_spacing = 1.3
index = np.arange(n_groups) * group_spacing

bar_width = 0.12
intra_pair_gap = 0.02
alg_gap = 0.10  # used between algorithm pairs AND before PointMap

pair_w = 2 * bar_width + intra_pair_gap

total_w = 3 * pair_w + 3 * alg_gap + bar_width

positions_raw = [
    0.0,  # PointPatch noFF
    0.0 + bar_width + intra_pair_gap,  # PointPatch FF
    pair_w + alg_gap,  # DP3 noFF
    pair_w + alg_gap + bar_width + intra_pair_gap,  # DP3 FF
    2 * pair_w + 2 * alg_gap,  # PCM noFF
    2 * pair_w + 2 * alg_gap + bar_width + intra_pair_gap,  # PCM FF
    3 * pair_w + 3 * alg_gap,  # PointMap
]

center = total_w / 2.0
offsets = np.array(positions_raw) - center

# -------------------------
# 3. PREPARE DATA FOR PGFPLOTS
# -------------------------

bars_data = []  # list of (label, x_positions, y_values)
for offs, data, lab in zip(offsets, data_arrays, labels_per_bar):
    xs = index + offs  # same spacing as you used for ax.bar
    bars_data.append((lab, xs, data))

max_val = max(max(d) for d in data_arrays)
ymin = -0.12 * max_val
ymax = 1.15 * max_val

# -------------------------
# 4. BUILD PGFPLOTS CODE
# -------------------------

lines = []

lines.append(r"\begin{tikzpicture}")
lines.append(r"\begin{axis}[")
lines.append(r"  ybar,")
# scale x so that 1 x-unit = 1cm; bar_width/intra_pair_gap/group_spacing
# are then in cm, preserving your relative spacing
lines.append(r"  x=1cm,")
lines.append(f"  bar width={bar_width:.3f}cm,")
lines.append(f"  ymin={ymin:.3f},")
lines.append(f"  ymax={ymax:.3f},")
lines.append(r"  width=10cm,")
lines.append(r"  height=6cm,")
lines.append(r"  xtick=\empty,")  # we place our own labels
lines.append(r"  ylabel={Score},")
lines.append(r"  ymajorgrids=true,")
lines.append(r"]")

# --- the bars themselves ---
for lab, xs, ys in bars_data:
    lines.append(r"\addplot coordinates {")
    for x, y in zip(xs, ys):
        lines.append(f"  ({x:.4f},{y:.4f})")
    lines.append(r"};")

# --- per-bar labels under each bar (like your ax.text below 0) ---
label_y = -0.05 * max_val
for lab, xs, ys in bars_data:
    for x in xs:
        # one label per bar at fixed y below zero
        lines.append(
            r"\node[anchor=north,font=\scriptsize] "
            f"at (axis cs:{x:.4f},{label_y:.4f}) {{{lab}}};"
        )

# --- group labels above each group center ---
group_label_y = 1.08 * max_val
for xg, gname in zip(index, group_labels):
    lines.append(
        r"\node[anchor=south,font=\bfseries] "
        f"at (axis cs:{xg:.4f},{group_label_y:.4f}) {{{gname}}};"
    )

lines.append(r"\end{axis}")
lines.append(r"\end{tikzpicture}")

# -------------------------
# 5. WRITE TO FILE
# -------------------------

with open("main_results_bars_ybar.tex", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
