import re

import matplot2tikz
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch


def plot_results():
    # Set style
    plt.style.use("default")
    sns.set_palette("husl")

    bar_width = 0.4

    # Example data - replace with your actual performance values
    methods = [
        # ==== Robocasa Group ====
        "PointPatch",
        "PointPatch + FF",
        "DP3",
        "DP3 + FF",
        "PCM",
        "PCM + FF",
        "PointMap",
        # ==== Maniskill group ===
        "PointPatch",
        "PointPatch + FF",
        "DP3",
        "DP3 + FF",
        "PCM",
        "PCM + FF",
        "PointMap",
        # ==== Real Robot Group ====
        "PointPatch",
        "PointPatch + FF",
    ]
    success_rates = [
        # ==== Robocasa Group ====
        0.210375,  # PointPatch
        0.3999375,  # PointPatch + FF
        0.190625,  # DP3
        0.2635,  # DP3 + FF
        0.7,  # PCM (not correct yet)
        0.7,  # PCM + FF (not correct yet)
        0.061375,  # PointMap
        # ==== Maniskill group ====
        0.508335,  # PointPatch
        0.5749975,  # PointPatch + FF
        0.5875025,  # DP3
        0.6416675,  # DP3 + FF
        0.620,  # PCM
        0.623,  # PCM + FF
        0.6083325,  # PointMap + FF
        # ==== Real Robot Group ====
        0.0521,  # PointPatch
        0.349,  # PointPatch + FF
    ]

    # Create custom x positions to group methods properly
    x_positions = [
        # ==== Robocasa Group ====
        0,  # PointPatch
        bar_width,  # PointPatch + FF
        3 * bar_width,  # DP3
        4 * bar_width,  # DP3 + FF
        6 * bar_width,  # PCM
        7 * bar_width,  # PCM + FF
        9 * bar_width,  # PointMap,
        # ==== Maniskill group ====
        12 * bar_width,  # PointPatch
        13 * bar_width,  # PointPatch + FF
        15 * bar_width,  # DP3
        16 * bar_width,  # DP3 + FF
        18 * bar_width,  # PCM
        19 * bar_width,  # PCM + FF
        21 * bar_width,  # PointMap
        # ==== Real Robot Group ====
        24 * bar_width,  # DP3
        25 * bar_width,  # DP3 + FF
    ]  # RGB alone, then paired methods

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 8))

    # Define colors: RGB alone, then pairs with similar colors
    colors = [
        # ==== Robocasa Group ====
        "#3498db",  # PointPatch (blue)
        "#2980b9",  # PointPatch + FF (darker blue)
        "#e74c3c",  # DP3 (red)
        "#c0392b",  # DP3 + FF (darker red)
        "#2ecc71",  # PCM (green)
        "#27ae60",  # PCM + FF (darker green)
        "#9b59b6",  # PointMap (purple, standalone)
        # ==== Maniskill group ====
        "#3498db",  # PointPatch (blue)
        "#2980b9",  # PointPatch + FF (darker blue)
        "#e74c3c",  # DP3 (red)
        "#c0392b",  # DP3 + FF (darker red)
        "#2ecc71",  # PCM (green)
        "#27ae60",  # PCM + FF (darker green)
        "#9b59b6",  # PointMap (purple, standalone)
        # ==== Real Robot Group ====
        "#e74c3c",  # DP3 (red)
        "#c0392b",  # DP3 + FF (darker red)
    ]

    greyer_colors = [
        # ==== Robocasa Group ====
        "#96a3ad",  # PointPatch (blue, tinted grey)
        "#2980b9",  # PointPatch + FF (blue)
        "#b8a3a0",  # DP3 (red, tinted grey)
        "#c0392b",  # DP3 + FF (red)
        "#a0b2a8",  # PCM (green, tinted grey)
        "#27ae60",  # PCM + FF (green)
        "#b8a8c5",  # PointMap (purple, tinted grey)
        # ==== Maniskill Group ====
        "#96a3ad",  # PointPatch (blue, tinted grey)
        "#2980b9",  # PointPatch + FF (blue)
        "#b8a3a0",  # DP3 (red, tinted grey)
        "#c0392b",  # DP3 + FF (red)
        "#a0b2a8",  # PCM (green, tinted grey)
        "#27ae60",  # PCM + FF (green)
        "#b8a8c5",  # PointMap (purple, tinted grey)
        # ==== Real Robot Group ====
        "#b8a3a0",  # DP3 (red, tinted grey)
        "#c0392b",  # DP3 + FF (red)
    ]

    hatches = [
        # ==== Robocasa Group ====
        "",  # PointPatch (blue)
        "/",  # PointPatch + FF (darker blue)
        "",  # DP3 (red)
        "/",  # DP3 + FF (darker red)
        "",  # PCM (green)
        "/",  # PCM + FF (darker green)
        "",  # PointMap (purple, standalone)
        # ==== Maniskill group ====
        "",  # PointPatch (blue)
        "/",  # PointPatch + FF (darker blue)
        "",  # DP3 (red)
        "/",  # DP3 + FF (darker red)
        "",  # PCM (green)
        "/",  # PCM + FF (darker green)
        "",  # PointMap (purple, standalone)
        # ==== Real Robot Group ====
        "",  # DP3 (red)
        "/",  # DP3 + FF (darker red)
    ]

    # Create bar chart with custom positions and slimmer bars
    bars = ax.bar(
        x_positions,
        success_rates,
        width=0.4,
        color=greyer_colors,
        alpha=0.8,
        edgecolor="white",
        linewidth=1.2,
        # hatch=hatches,
    )

    # Group label settings
    y_max = max(success_rates)
    group_name_position = y_max + 0.04
    group_name_font_size = 20

    # Add group labels
    # Group A: Robocasa
    x_center_1 = (0 + 9 * bar_width) / 2
    ax.text(
        x_center_1,  # x position
        group_name_position,  # y position slightly above tallest bar
        "Robocasa",
        ha="center",
        va="bottom",
        fontsize=group_name_font_size,
        fontweight="bold",
    )

    # Add separator line between groups
    ax.axvline(x=10.5 * bar_width, color="black", linestyle="--", alpha=0.7)

    # Group B: Maniskill
    x_center_2 = (12 * bar_width + 21 * bar_width) / 2
    ax.text(
        x_center_2,  # x position
        group_name_position,  # y position slightly above tallest bar
        "Maniskill",
        ha="center",
        va="bottom",
        fontsize=group_name_font_size,
        fontweight="bold",
    )

    # Add separator line between groups
    ax.axvline(x=22.5 * bar_width, color="black", linestyle="--", alpha=0.7)

    # Group C: Real Robot
    x_center_3 = (24 * bar_width + 25 * bar_width) / 2
    ax.text(
        x_center_3 + 0.1,  # x position (slightly adjusted to the right to center)
        group_name_position,  # y position slightly above tallest bar
        "Real",
        ha="center",
        va="bottom",
        fontsize=group_name_font_size,
        fontweight="bold",
    )

    # Add title
    # ax.set_title(
    #     "Mean Success Rate: Leg Lifted", fontsize=27, fontweight="bold", pad=20
    # )

    # Customize the chart with larger fonts
    # ax.set_xlabel("Method", fontsize=22, fontweight="bold")
    ax.set_ylabel("Mean Success Rate", fontsize=12, fontweight="bold")

    # Set y-axis to show percentages
    ax.set_ylim(0, 0.799)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0%}".format(y)))

    # Increase tick label font sizes
    ax.tick_params(axis="x", labelsize=14)  # Reduced slightly to fit longer labels
    ax.tick_params(axis="y", labelsize=14)

    # Set custom x-axis labels and positions
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        methods, rotation=45, ha="right", fontsize=8
    )  # Slight rotation for better fit

    # Add value labels on top of bars with larger font
    # for pos, rate in zip(x_positions, success_rates):
    #     ax.text(
    #         pos,
    #         rate + 0.01,
    #         f"{rate:.1%}",
    #         ha="center",
    #         va="bottom",
    #         fontweight="bold",
    #         fontsize=14,
    #     )

    # Add subtle grid
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Group related methods with subtle background rectangles
    ax.axvspan(-0.3, 0.3, alpha=0.1, color="purple", zorder=0)  # RGB-Only
    ax.axvspan(0.9, 2.1, alpha=0.1, color="blue", zorder=0)  # RGB-D methods
    ax.axvspan(2.7, 3.9, alpha=0.1, color="red", zorder=0)  # PointMap methods
    ax.axvspan(4.5, 5.7, alpha=0.1, color="green", zorder=0)  # PointPatch methods

    # Create custom legend
    legend_elements = [
        Patch(facecolor="white", edgecolor="black", hatch="//", label="Hatched"),
        Patch(facecolor="white", edgecolor="black", label="Not hatched"),
    ]

    ax.legend(handles=legend_elements, title="Bar Types", loc="upper right")

    # plt.show()

    # Improve layout
    tikz_code = matplot2tikz.get_tikz_code(axis_width="14cm", axis_height="8cm")

    tikz_code = re.sub(r"(?<!\\)%\}", "\%}", tikz_code)

    with open("main_results.tex", "w") as f:
        f.write(tikz_code)

    # plt.tight_layout()
    # plt.savefig(
    #     "/Users/enricokrohmer/Desktop/main_results.pdf",
    #     format="pdf",
    #     dpi=300,
    #     bbox_inches="tight",
    # )
    # plt.show()


def plot_sin():
    # Generate x values from 0 to 2π
    x = np.linspace(0, 2 * np.pi, 1000)

    # Number of subplots (different frequencies)
    frequencies = [2, 4]
    num_plots = len(frequencies)

    # Create figure with multiple subplots stacked vertically
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 10))
    fig.patch.set_facecolor("#E1D5E7")

    # If only one subplot, axes won't be an array
    if num_plots == 1:
        axes = [axes]

    frequencies = list(reversed(frequencies))
    for i, freq in enumerate(frequencies):
        ax = axes[i]

        # Calculate sine wave with current frequency
        y = np.sin(freq * x)

        # Set background color for each subplot
        ax.set_facecolor("#E1D5E7")

        # Plot sine curve
        ax.plot(x, y, "black", linewidth=2)

        # Customize the coordinate system
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)  # x-axis
        ax.axvline(x=0, color="black", linestyle="-", alpha=0.3)  # y-axis

        # Remove all ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])

        # Remove axis spines
        ax.spines["bottom"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Set y-axis limits
        ax.set_ylim(-1.2, 1.2)

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.1)

    # Save to PDF
    pdf_filename = "stacked_sine_curves.pdf"
    with PdfPages(pdf_filename) as pdf:
        pdf.savefig(fig, facecolor="#E1D5E7", edgecolor="none")

    plt.tight_layout()
    print(f"PDF saved as: {pdf_filename}")

    # Optional: Display the plot
    plt.show()

    plt.close()


def plot_sin2():
    # Set up the figure and subplots
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    # Define the two points to track
    x1, x2 = 2.0, 2.5  # Close points in original space
    x = np.linspace(0, 8, 1000)

    # Define frequencies for each subplot
    frequencies = [1, 2, 4, 8]
    labels = ["f = 1 (Original)", "f = 2", "f = 4", "f = 8"]

    for i, (freq, label) in enumerate(zip(frequencies, labels)):
        ax = axes[i]

        # Plot the sinusoid
        y = np.sin(freq * x)
        ax.plot(x, y, "b-", linewidth=2, alpha=0.7)

        # Calculate y-values for our two points
        y1 = np.sin(freq * x1)
        y2 = np.sin(freq * x2)

        # Mark the two points
        ax.plot(x1, y1, "ro", markersize=8, label=f"Point 1: ({x1:.1f}, {y1:.2f})")
        ax.plot(x2, y2, "go", markersize=8, label=f"Point 2: ({x2:.1f}, {y2:.2f})")

        # Draw vertical lines through the points
        ax.axvline(x=x1, color="red", linestyle="--", alpha=0.6, linewidth=2)
        ax.axvline(x=x2, color="green", linestyle="--", alpha=0.6, linewidth=2)

        # Calculate and display the distance between mapped points
        distance = abs(y2 - y1)
        ax.text(
            0.02,
            0.95,
            f"{label}",
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
        )
        ax.text(
            0.02,
            0.80,
            f"Distance: {distance:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7),
        )

        # Set axis properties
        ax.set_xlim(0, 8)
        ax.set_ylim(-1.2, 1.2)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

        # Only show x-label on bottom plot
        if i == len(axes) - 1:
            ax.set_xlabel("x (Cartesian coordinate)", fontsize=12)
        ax.set_ylabel(f"sin({freq}x)", fontsize=12)

    plt.tight_layout()
    plt.show()

    pdf_filename = "stacked_sine_curves.pdf"
    with PdfPages(pdf_filename) as pdf:
        pdf.savefig(fig, facecolor="#FFFFFF", edgecolor="none")


if __name__ == "__main__":
    plot_results()
