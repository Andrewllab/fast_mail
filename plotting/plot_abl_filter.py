import re

import matplotlib.pyplot as plt
import numpy as np
import yaml


def visualize_results(yaml_file_path):
    with open(yaml_file_path, "r") as f:
        data = yaml.safe_load(f)

    # Dictionary to store structured data:
    # structured_data[environment][method] = { 'sigmas': [], 'means': [], 'bcis': [] }
    structured_data = {}

    # Regular expression to extract the float sigma value from the key string
    sigma_regex = re.compile(r"t10_jitter\.sigma=([\d\.]+)")

    for method_key, env_dict in data.items():
        # Determine method label and boolean for styling
        fourier_label = (
            "Fourier"
            if "spatial_encoder._partial_=True" in method_key
            else "w/o Fourier"
        )
        # Extract numerical sigma
        match = sigma_regex.search(method_key)
        sigma = float(match.group(1))

        for env_name, metrics in env_dict.items():
            if env_name not in structured_data:
                structured_data[env_name] = {}
            if fourier_label not in structured_data[env_name]:
                structured_data[env_name][fourier_label] = {
                    "sigmas": [],
                    "means": [],
                    "bcis": [],
                }

            structured_data[env_name][fourier_label]["sigmas"].append(sigma)
            structured_data[env_name][fourier_label]["means"].append(metrics["mean"])
            structured_data[env_name][fourier_label]["bcis"].append(metrics["bci"])

    # Setup Plot
    plt.figure(figsize=(12, 8))

    # Use a colormap to get a unique color for each environment
    envs = sorted(list(structured_data.keys()))
    colors = plt.cm.tab20(np.linspace(0, 1, len(envs)))
    env_to_color = dict(zip(envs, colors))

    for env_name in envs:
        color = env_to_color[env_name]

        # norm = max(structured_data[env_name]["Fourier"]["means"])
        norm = structured_data[env_name]["Fourier"]["means"][0]
        norm = max(norm, 0.1)  # Avoid division by zero

        for method_label, vals in structured_data[env_name].items():
            # if method_label != "Fourier":
            #     continue  # Only plot Fourier for now, as per the original code's emphasis

            # Sort data by sigma to ensure lines connect correctly
            sorted_indices = np.argsort(vals["sigmas"])
            sigmas = np.array(vals["sigmas"])[sorted_indices]
            means = np.array(vals["means"])[sorted_indices] / norm
            bcis = np.array(vals["bcis"])[sorted_indices] / norm

            # Style: Fourier Features are thicker
            linewidth = 3.5 if method_label == "Fourier" else 1.2
            # Use alpha for transparency in the shaded regions
            alpha = 0.15 if method_label == "Fourier" else 0.08

            # Plot the mean line
            # Label only once per environment/method combo for the legend
            line_label = f"{env_name} ({method_label})"
            plt.plot(sigmas, means, color=color, linewidth=linewidth, label=line_label)

            # Shade the confidence interval (mean +/- bci)
            # plt.fill_between(
            #     sigmas, means - bcis, means + bcis, color=color, alpha=alpha
            # )

    # Formatting
    plt.xscale("log")  # Often noise levels are better viewed on a log scale
    plt.xlabel("Noise Magnitude (Sigma)", fontsize=12)
    plt.ylabel("Success Rate", fontsize=12)
    plt.title("Environment Success Rate vs. Noise Magnitude", fontsize=14)

    # Place legend outside because there will be many lines
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small", ncol=2)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()

    plt.savefig("noise_robustness_plot.png")
    plt.show()


if __name__ == "__main__":
    # Ensure you have the 'PyYAML' library installed: pip install pyyaml
    visualize_results("stats/2026-03-30_16-49-19/results.yaml")
