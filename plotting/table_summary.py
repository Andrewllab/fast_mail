import re
from pathlib import Path

import hydra
import pandas as pd
import yaml
from omegaconf import DictConfig
from rootutils import rootutils

# enables importing local modules regardless of where the script is run
rootutils.setup_root(__file__, indicator=".isort.cfg", pythonpath=True)

FOLDERNAME = "plotting/stats/2026-04-04_23-27-58"

TASK_SUITES = {
    "RoboCasa": "roca",
    "ManiSkill3": "ms",
}

LABELS = {
    "PointPatch": ("main_%s_pp", None),
    "PointPatch-attn": ("main_%s_pp+attn", None),
    "PCM": ("main_%s_pcm", None),
    "DP3": ("main_%s_dp3", None),
    "PointTransformer": ("main_%s_pointtransformer", None),
    "PointPatch+RGB": ("main_%s_pp+rgb", "encoder.pretrained=False"),
    "PointPatch+RGB (pretrained)": ("main_%s_pp+rgb", "encoder.pretrained=True"),
    # "Depth": ("main_%s_depth", None),
    "PointMap": ("main_%s_pmp", None),
    "RGB": ("main_%s_rgb", None),
}


@hydra.main(version_base=None, config_path="conf", config_name="tables")
def main(cfg: DictConfig) -> None:

    stats_dir = Path(cfg.paths.root_dir) / FOLDERNAME
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # organize by parent folder name, which contains the tag
    # get the tag from the foldername by stripping "series-" from the start
    stats_files = {
        path.parent.name[7:]: path for path in stats_dir.glob("**/bci_stats.yaml")
    }

    for task_suite, task_suite_short in TASK_SUITES.items():

        table_data = []
        for label, (tag, filter) in LABELS.items():
            tag = tag % task_suite_short
            stats_file = stats_files[tag]

            # Load YAML files
            with open(stats_file, "r") as f:
                stats = yaml.safe_load(f)

            if filter is not None:
                stats = {
                    key: value
                    for key, value in stats.items()
                    if re.search(filter, key) is not None
                }

            if label in ["Depth", "PointMap", "RGB"]:
                # these baselines don't have a +FF version
                assert list(stats.keys()) == ["noop=None"]

                stats[False] = stats.pop("noop=None")

            else:
                assert len(stats) == 2

                # stats for each tag should contain one entry with FF and one without
                group_name_ff = [
                    key
                    for key in stats.keys()
                    if "spatial_encoder._partial_=True" in key
                ]
                assert len(group_name_ff) == 1
                group_name_ff = group_name_ff[0]

                group_name_no_ff = [
                    key
                    for key in stats.keys()
                    if "spatial_encoder._partial_=None" in key
                ]
                assert len(group_name_no_ff) == 1
                group_name_no_ff = group_name_no_ff[0]

                stats[True] = stats.pop(group_name_ff)
                stats[False] = stats.pop(group_name_no_ff)

            # just get the overall stats, which are averaged across environments
            stats = {
                group_name: stats_by_env for group_name, stats_by_env in stats.items()
            }

            # count statistically significant improvements
            if True in stats:
                improved = [
                    env
                    for env in stats[True]
                    if (stats[True][env]["mean"] - stats[False][env]["mean"])
                    > (stats[True][env]["bci"] + stats[False][env]["bci"])
                ]
            else:
                improved = None

            table_data.append(
                {
                    task_suite: label,
                    "baseline (% success)": f"{stats[False]['overall']['mean'] * 100:.1f} ± {stats[False]['overall']['bci'] * 100:.1f}",
                    "+ FF (% success)": (
                        f"{stats[True]['overall']['mean'] * 100:.1f} ± {stats[True]['overall']['bci'] * 100:.1f}"
                        if True in stats
                        else "-"
                    ),
                    "# improved": f"{len(improved)}" if improved is not None else -",
                }
            )

        # Create DataFrame
        df = pd.DataFrame(table_data)

        # Format for clean display
        print(df.to_string(index=False, float_format=lambda x: "{:.3f}".format(x)))

        # Save to markdown
        df.to_markdown(
            output_dir / f"summary_{task_suite_short}.md",
            index=False,
            floatfmt=(None, ".1f", ".2f", ".2f"),
        )


if __name__ == "__main__":
    main()  # type: ignore[misc]
