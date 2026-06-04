import re
from pathlib import Path

import hydra
import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    version_base=None, config_path="conf", config_name="table_main_pointtransformer"
)
def main(cfg: DictConfig) -> None:

    results_file = Path(cfg.results_file)

    # Load YAML file
    with open(results_file, "r") as f:
        results = yaml.safe_load(f)

    results_by_env = {}
    for method_key, env_dict in results.items():
        # Determine method label and boolean for styling
        fourier_label = (
            "Fourier"
            if "spatial_encoder._partial_=True" in method_key
            else "w/o Fourier"
        )

        for env_name, metrics in env_dict.items():
            if env_name not in results_by_env:
                results_by_env[env_name] = {}

            results_by_env[env_name][fourier_label] = {
                "mean": metrics["mean"],
                "bci": metrics["bci"],
            }

    table_data = []

    for env in results_by_env.keys():
        table_data.append(
            {
                "Environment": env,
                "PointTransformer w/ FF (% success)": f"{results_by_env[env]['Fourier']['mean'] * 100:.1f} ± {results_by_env[env]['Fourier']['bci'] * 100:.1f}",
                "PointTransformer (% success)": f"{results_by_env[env]['w/o Fourier']['mean'] * 100:.1f} ± {results_by_env[env]['w/o Fourier']['bci'] * 100:.1f}",
            }
        )

    # Create DataFrame
    df = pd.DataFrame(table_data)

    # Format for clean display
    print(df.to_string(index=False, float_format=lambda x: "{:.3f}".format(x)))

    # Save to markdown
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    df.to_markdown(
        output_dir / "main_pointtransformer.md",
        index=False,
    )


if __name__ == "__main__":
    main()  # type: ignore[misc]
