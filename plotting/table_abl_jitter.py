import re
from pathlib import Path

import hydra
import pandas as pd
import yaml
from omegaconf import DictConfig
from rootutils import rootutils

# enables importing local modules regardless of where the script is run
rootutils.setup_root(__file__, indicator=".isort.cfg", pythonpath=True)

FOLDERNAME = "plotting/stats/2026-04-07_15-28-17"
FOURIER_LABELS = {
    "True": "FFs",
    "None": "no FFs",
}
JITTER_LABELS = {
    "None": "no jitter",
    "uniform jitter": "random jitter",
    "VariableJitter": "VariableJitter",
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
        match = re.search(r"t11_tokenizer\.spatial_encoder\._partial_=([a-zA-Z]+)", key)
        assert match

        fourier_label = FOURIER_LABELS[match.group(1)]

        match = re.search(r"t10_jitter\.name=([a-zA-Z\s]+)", key)
        assert match

        jitter_label = JITTER_LABELS[match.group(1)]

        # just get the overall stats, which are averaged across environments
        stats[(fourier_label, jitter_label)] = stats.pop(key)["overall"]

    table_data = []
    table_data.append(
        {
            "method": "ours",
            "success rate (%)": f"{stats[('FFs', 'VariableJitter')]['mean'] * 100:.1f} ± {stats[('FFs', 'VariableJitter')]['bci'] * 100:.1f}",
        }
    )

    for label in JITTER_LABELS.values():
        table_data.append(
            {
                "method": f"no FFs, {label}",
                "success rate (%)": f"{stats[('no FFs', label)]['mean'] * 100:.1f} ± {stats[('no FFs', label)]['bci'] * 100:.1f}",
            }
        )

    for label in JITTER_LABELS.values():
        if label == "VariableJitter":
            # this is the same as ours, so skip it
            continue

        table_data.append(
            {
                "method": f"FFs, {label}",
                "success rate (%)": f"{stats[('FFs', label)]['mean'] * 100:.1f} ± {stats[('FFs', label)]['bci'] * 100:.1f}",
            }
        )

    # Create DataFrame
    df = pd.DataFrame(table_data)

    # Format for clean display
    print(df.to_string(index=False, float_format=lambda x: "{:.3f}".format(x)))

    # Save to markdown
    df.to_markdown(output_dir / f"abl_jitter.md", index=False)


if __name__ == "__main__":
    main()  # type: ignore[misc]
