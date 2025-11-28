import re

import matplot2tikz
import matplotlib.pyplot as plt
import numpy as np


def make_bar(
    ax,
    labels,
    vals,
    title,
    colors,
    yerr=None,
    bar_width=0.8,
    xscale=1.0,
    keep_ylabel=True,
):
    # xscale lets you pack bars closer or further apart
    x = np.arange(len(labels)) * xscale

    if yerr is not None:
        yerr_array = np.array([e if e is not None else 0.0 for e in yerr])
        ax.bar(
            x,
            vals,
            width=bar_width,
            color=colors,
            # edgecolor="black",
            yerr=yerr_array,
            capsize=2,
        )
    else:
        ax.bar(x, vals, width=bar_width, color=colors, edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=20)
    ax.set_title(title, fontsize=24)
    if keep_ylabel:
        ax.set_ylabel("Mean Success Rate", fontsize=24)
    ax.grid(axis="y", linestyle=":", linewidth=0.5)
    ax.tick_params(labelleft=True)
    # increase tick label size (both axes)
    ax.tick_params(axis="both", labelsize=20)


highlight_color = "#2980b9"
ablations_grey = "#96a3ad"


# # Data for Jitter + Fourier Features ablations
labels1 = [
    "no FFs, no jitter",
    "no FFs, random jitter",
    "no FFs, VariableJitter",
    "FFs, no jitter",
    "FFs, random jitter",
    "FFs, VariableJitter",
]
vals1 = [
    0.1729,
    0.1729,
    0.1625,
    0.3854,
    0.3854,
    0.3812,
]
std1 = [
    0.0201,
    0.0321,
    0.0,
    0.0295,
    0.0315,
    0.0062,
]
colors1 = [
    highlight_color if lab == "FFs, VariableJitter" else ablations_grey
    for lab in labels1
]


# Data for Wavelengths ablations
labels2 = [
    "logsp(4, 0.03, 8)",
    "logsp(4, 0.06, 8)",
    "logsp(4, 0.1, 8)",
    "logsp(4, 0.2, 8)",
    "logsp(4, 0.4, 8)",
    "logsp(4, 0.03, 4)",
    "logsp(4, 0.06, 4)",
    "logsp(4, 0.1, 4)",
    "logsp(4, 0.2, 4)",
    "logsp(4, 0.4, 4)",
]
vals2 = [
    0.4083,
    0.3812,
    0.4146,
    0.3688,
    0.3542,
    0.3604,
    0.3458,
    0.3625,
    0.3458,
    0.3917,
]
std2 = [
    0.0308,
    0.0062,
    0.013,
    0.0108,
    0.0377,
    0.0095,
    0.0581,
    0.0165,
    0.0397,
    0.0219,
]
colors2 = [highlight_color if "0.06, 8" in lab else ablations_grey for lab in labels2]


# # Data for Learnable Fourier Features ablations
labels3 = [
    "logsp",
    "logsp + SPE",
    "RFF",
    "RFF + learned",
    "RFF + SPE",
    "RFF + Cartesian",
]
vals3 = [
    0.3812,
    0.3646,
    0.2417,
    0.2562,
    0.2229,
    0.0,
]
std3 = [
    0.0062,
    0.0191,
    0.0144,
    0.0331,
    0.0308,
    0.0,
]
colors3 = [highlight_color if lab == "logsp" else ablations_grey for lab in labels3]


if __name__ == "__main__":
    # adapt figure size based on number of bars
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(6, 4),
        sharey=True,
        # gridspec_kw={"width_ratios": [len(labels1) * 4, len(labels2) * 4, len(labels3) * 4]}
        # gridspec_kw={"width_ratios": [len(labels1) * 4, len(labels3) * 4]}
        gridspec_kw={"width_ratios": [len(labels1), len(labels2), len(labels3)]},
    )
    # same bar width for all plots (in data units)
    bar_width = 0.8

    make_bar(
        axes[0],
        labels1,
        vals1,
        "Jitter Augmentation",
        colors1,
        yerr=std1,
        bar_width=bar_width,
        keep_ylabel=True,
    )
    make_bar(
        axes[1],
        labels2,
        vals2,
        "Wavelength Configuration",
        colors2,
        yerr=std2,
        bar_width=bar_width,
        keep_ylabel=False,
    )
    make_bar(
        axes[2],
        labels3,
        vals3,
        "Random and Learnable Features",
        colors3,
        yerr=std3,
        bar_width=bar_width,
        keep_ylabel=False,
    )

    # adapt y-limits
    for ax in axes:
        ax.set_ylim(0, 0.5)

    plt.subplots_adjust(bottom=0.35, wspace=0.35)

    # fig, ax = plt.subplots(figsize=(8, 4))  # adjust this width to tune spacing

    # make_bar(
    #     ax, labels3, vals3, "Learnable FF", colors3, bar_width=0.8, xscale=1.0
    # )  # xscale<1 -> bars closer; >1 -> further

    # ax.set_ylim(0, 0.5)

    # plt.subplots_adjust(bottom=0.35)

    # plt.show()

    tikz_code = matplot2tikz.get_tikz_code(
        axis_width="99cm", axis_height="6cm", textsize=10
    )

    # fix widths of subplots, since matplot2tikz does not handle width_ratios correctly
    tikz_code = tikz_code.replace("width=99cm", "width=8.59cm", 1)
    tikz_code = tikz_code.replace("width=99cm", "width=14.31cm", 1)
    tikz_code = tikz_code.replace("width=99cm", "width=8.59cm", 1)

    tikz_code = re.sub(r"(?<!\\)%\}", r"\%}", tikz_code)

    with open("ablation_results.tex", "w") as f:
        f.write(tikz_code)
