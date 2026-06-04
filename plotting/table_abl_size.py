import re
from pathlib import Path

import hydra
import pandas as pd
import yaml
from omegaconf import DictConfig

PCD_SIZES = {
    0.01: "18k",
    0.015: "15k",
    0.02: "9k",
    0.03: "4.5k",
    0.04: "3k",
    0.05: "2k",
}


@hydra.main(version_base=None, config_path="conf", config_name="table_abl_size")
def main(cfg: DictConfig) -> None:

    # Load YAML files
    with open(cfg.results_file, "r") as f:
        results = yaml.safe_load(f)

    # Regular expression to extract the float sigma value from the key string
    voxel_size_regex = re.compile(r"t2_sample_voxel\.size=([\d\.]+)")

    results_by_sigma = {}
    for method_key, env_dict in results.items():
        # Determine method label and boolean for styling
        fourier_label = (
            "Fourier"
            if "spatial_encoder._partial_=True" in method_key
            else "w/o Fourier"
        )
        # Extract numerical size
        match = voxel_size_regex.search(method_key)
        size = float(match.group(1))

        if size not in results_by_sigma:
            results_by_sigma[size] = {}

        results_by_sigma[size][fourier_label] = {
            "mean": env_dict["overall"]["mean"],
            "bci": env_dict["overall"]["bci"],
        }

    table_data = []

    for size in results_by_sigma.keys():
        table_data.append(
            {
                "voxel size (cm)": size * 100,
                "PointPatch w/ FF (% success)": f"{results_by_sigma[size]['Fourier']['mean'] * 100:.1f} ± {results_by_sigma[size]['Fourier']['bci'] * 100:.1f}",
                "PointPatch (% success)": f"{results_by_sigma[size]['w/o Fourier']['mean'] * 100:.1f} ± {results_by_sigma[size]['w/o Fourier']['bci'] * 100:.1f}",
                "Approx. Point Count": PCD_SIZES.get(size, "N/A"),
            }
        )

    # Create DataFrame
    df = pd.DataFrame(table_data)

    # Sort by the second column (Relative Success Max Jitter)
    df = df.sort_values(by="voxel size (cm)", ascending=True)

    # Format for clean display
    print(df.to_string(index=False, float_format=lambda x: "{:.3f}".format(x)))

    # Save to markdown
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    df.to_markdown(
        output_dir / "abl_size.md",
        index=False,
        floatfmt=(None, ".1f", ".2f", ".2f"),
    )


if __name__ == "__main__":
    main()  # type: ignore[misc]
