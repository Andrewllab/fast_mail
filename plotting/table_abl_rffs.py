import re
from pathlib import Path

import hydra
import pandas as pd
import yaml
from omegaconf import DictConfig
from rootutils import rootutils

# enables importing local modules regardless of where the script is run
rootutils.setup_root(__file__, indicator=".isort.cfg", pythonpath=True)

FOLDERNAME = "plotting/stats/2026-04-07_19-22-17"

LABELS = {
    ("FourierFeatures", "False"): "Ours",
    ("FourierFeatures", "True"): "log-spaced + SPE",
    ("Gaussian RFFs", "False"): "RFF",
    ("Gaussian RFFs + learned", "False"): "RFF + learned",
    ("Gaussian RFFs + SPE", "True"): "RFF + SPE",
    ("Gaussian RFFs + identity", "False"): "RFF + Cartesian",
}


@hydra.main(version_base=None, config_path="conf", config_name="tables")
def main(cfg: DictConfig) -> None:

    stats_dir = Path(cfg.paths.root_dir) / FOLDERNAME
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # organize by parent folder name, which contains the tag
    # get the tag from the foldername by stripping "series-" from the start
    stats_file = stats_dir / "bci_stats.yaml"

    # Load YAML file
    with open(stats_file, "r") as f:
        stats = yaml.safe_load(f)

    for key in list(stats.keys()):
        match_name = re.search(
            r"t11_tokenizer\.spatial_encoder\.name=([a-zA-Z\s+]+)", key
        )
        assert match_name

        match_spe = re.search(
            r"t11_tokenizer\.spatial_encoder\.linear_transform=([a-zA-Z]+)", key
        )
        assert match_spe

        label = LABELS[(match_name.group(1), match_spe.group(1))]

        # just get the overall stats, which are averaged across environments
        stats[label] = stats.pop(key)["overall"]

    table_data = []

    for label in LABELS.values():

        table_data.append(
            {
                "method": label,
                "success rate (%)": f"{stats[label]['mean'] * 100:.1f} ± {stats[label]['bci'] * 100:.1f}",
            }
        )

    # Create DataFrame
    df = pd.DataFrame(table_data)

    # Format for clean display
    print(df.to_string(index=False, float_format=lambda x: "{:.3f}".format(x)))

    # Save to markdown
    df.to_markdown(output_dir / f"abl_rffs.md", index=False)


if __name__ == "__main__":
    main()  # type: ignore[misc]
