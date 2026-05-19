import re
from pathlib import Path

import hydra
import pandas as pd
import yaml
from omegaconf import DictConfig


def get_max_sigma_key(noise_dict):
    """Helper to find the key with the largest numerical sigma value."""
    sigma_regex = re.compile(r"sigma=([\d\.]+)")
    max_sigma = -1
    max_key = None
    for key in noise_dict.keys():
        match = sigma_regex.search(key)
        if match:
            val = float(match.group(1))
            if val > max_sigma:
                max_sigma = val
                max_key = key
    return max_key


@hydra.main(version_base=None, config_path="conf", config_name="table_abl_filter")
def main(cfg: DictConfig) -> None:

    # Load YAML files
    with open(cfg.baseline_yaml, "r") as f:
        base_data_raw = yaml.safe_load(f)

    baseline_data = {}
    for method_key, env_dict in base_data_raw.items():
        # Determine method label and boolean for styling
        fourier_label = (
            "Fourier"
            if "spatial_encoder._partial_=True" in method_key
            else "w/o Fourier"
        )

        for env_name, metrics in env_dict.items():
            if env_name not in baseline_data:
                baseline_data[env_name] = {}

            baseline_data[env_name][fourier_label] = {
                "mean": metrics["mean"],
                "bci": metrics["bci"],
            }

    # Regular expression to extract the float sigma value from the key string
    sigma_regex = re.compile(r"t10_jitter\.sigma=([\d\.]+)")

    with open(cfg.jitter_yaml, "r") as f:
        jitter_data_raw = yaml.safe_load(f)

    jitter_data = {}
    for method_key, env_dict in jitter_data_raw.items():
        fourier_label = (
            "Fourier"
            if "spatial_encoder._partial_=True" in method_key
            else "w/o Fourier"
        )
        # Extract numerical sigma
        match = sigma_regex.search(method_key)
        sigma = float(match.group(1))

        for env_name, metrics in env_dict.items():
            if env_name not in jitter_data:
                jitter_data[env_name] = {}

            if fourier_label not in jitter_data[env_name]:
                jitter_data[env_name][fourier_label] = {}

            jitter_data[env_name][fourier_label][sigma] = {
                "mean": metrics["mean"],
                "bci": metrics["bci"],
            }

    table_data = []

    for env in baseline_data.keys():
        # Extract means
        base_val = baseline_data[env]["Fourier"]["mean"]
        max_sigma = max(jitter_data[env]["Fourier"].keys())
        jit_val = jitter_data[env]["Fourier"][max_sigma]["mean"]
        no_f_val = baseline_data[env]["w/o Fourier"]["mean"]

        # Calculate Relative Success Rates
        # (Handling division by zero just in case baseline is 0.0)
        rel_jitter = jit_val / base_val if base_val > 0 else 0.0
        rel_no_fourier = no_f_val / base_val if base_val > 0 else 0.0

        table_data.append(
            {
                "": env,
                r"PointPatch w/ FF (% success)": base_val * 100,
                "+sigma=0.05 (relative)": rel_jitter,
                "-FF (relative)": rel_no_fourier,
            }
        )

    # Create DataFrame
    df = pd.DataFrame(table_data)

    # Sort by the second column (Relative Success Max Jitter)
    df = df.sort_values(by="+sigma=0.05 (relative)", ascending=True)

    # Format for clean display
    print(df.to_string(index=False, float_format=lambda x: "{:.3f}".format(x)))

    # Save to markdown
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    df.to_markdown(
        output_dir / "abl_filter.md",
        index=False,
        floatfmt=(None, ".1f", ".2f", ".2f"),
    )


if __name__ == "__main__":
    main()  # type: ignore[misc]
