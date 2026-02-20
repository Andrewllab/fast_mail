#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplot2tikz
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# 1. DATA (replace values)
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
# 2. LAYOUT / SPACING
# -------------------------

n_groups = len(group_labels)

# distance between group centers (controls spacing between groups)
group_spacing = 1.3
index = np.arange(n_groups) * group_spacing

bar_width = 0.12
intra_pair_gap = 0.02
alg_gap = 0.10  # used between algorithm pairs AND before PointMap

# width of one algorithm pair (two bars + internal gap)
pair_w = 2 * bar_width + intra_pair_gap

# total width = 3 pairs + 3 alg_gaps (before DP3, before PCM, before PointMap) + PointMap bar
total_w = 3 * pair_w + 3 * alg_gap + bar_width

# raw positions from left to right inside a group
positions_raw = [
    0.0,  # PointPatch noFF
    0.0 + bar_width + intra_pair_gap,  # PointPatch FF
    pair_w + alg_gap,  # DP3 noFF
    pair_w + alg_gap + bar_width + intra_pair_gap,  # DP3 FF
    2 * pair_w + 2 * alg_gap,  # PCM noFF
    2 * pair_w + 2 * alg_gap + bar_width + intra_pair_gap,  # PCM FF
    3 * pair_w + 3 * alg_gap,  # PointMap noFF (with full alg_gap before)
]

# center around 0 so the group is symmetric around its index position
center = total_w / 2.0
offsets = np.array(positions_raw) - center

# -------------------------
# 3. PLOT
# -------------------------

fig, ax = plt.subplots(figsize=(10, 3))

bars = []
for offs, data, lab in zip(offsets, data_arrays, labels_per_bar):
    b = ax.bar(index + offs, data, width=bar_width)
    bars.append((b, lab))

max_val = max(max(d) for d in data_arrays)

# -------------------------
# 4. LABEL UNDER EACH BAR
# -------------------------

for b, lab in bars:
    for rect in b:
        x = rect.get_x() + rect.get_width() / 2.0
        ax.text(
            x,
            -0.05 * max_val,  # a bit below 0
            lab,
            ha="center",
            va="top",
            rotation=0,
            fontsize=8,
        )

# extra bottom margin to fit rotated labels
ax.set_ylim(bottom=-0.12 * max_val, top=max_val * 1.15)

# -------------------------
# 5. GROUP LABEL ABOVE EACH GROUP
# -------------------------

for i, gname in enumerate(group_labels):
    ax.text(
        index[i],
        max_val * 1.08,
        gname,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

# -------------------------
# 6. AXES / GRID
# -------------------------

ax.set_ylabel("Score")
ax.set_title("Main Results")

ax.set_xticks([])  # no default x ticks; we use our own labels
ax.grid(axis="y", linestyle="--", linewidth=0.5)

# -------------------------
# 7. EXPORT TO TIKZ
# -------------------------


# plt.show()
matplot2tikz.clean_figure()
matplot2tikz.save("main_results_bars.tex", axis_width="10cm", axis_height="6cm")

plt.close(fig)
