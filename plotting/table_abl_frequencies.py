import re
from pathlib import Path

import hydra
import pandas as pd
import yaml
from omegaconf import DictConfig
from rootutils import rootutils

# enables importing local modules regardless of where the script is run
rootutils.setup_root(__file__, indicator=".isort.cfg", pythonpath=True)

FOLDERNAME = "plotting/stats/2026-04-07_13-42-42"

OURS_MIN_WAVELENGTH = 0.02
OURS_N_WAVELENGTHS = 16


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

    min_wavelengths = set()
    n_wavelengthses = set()

    # parse the keys to extract the min_wavelength and n_wavelengths values
    for key in list(stats.keys()):
        match = re.search(
            r"t11_tokenizer\.spatial_encoder\.min_wavelength=([\d\.]+)", key
        )
        assert match

        min_wavelength = float(match.group(1))
        min_wavelengths.add(min_wavelength)

        match = re.search(
            r"t11_tokenizer\.spatial_encoder\.n_wavelengths=([\d\.]+)", key
        )
        assert match

        n_wavelengths = int(match.group(1))
        n_wavelengthses.add(n_wavelengths)

        # just get the overall stats, which are averaged across environments
        stats[(min_wavelength, n_wavelengths)] = stats.pop(key)["overall"]

    min_wavelengths = sorted(min_wavelengths)
    n_wavelengthses = sorted(n_wavelengthses)

    table_data = []
    table_data.append(
        {
            "method": f"ours (L={OURS_N_WAVELENGTHS}, λ_min={OURS_MIN_WAVELENGTH})",
            "success rate (%)": f"{stats[(OURS_MIN_WAVELENGTH, OURS_N_WAVELENGTHS)]['mean'] * 100:.1f} ± {stats[(OURS_MIN_WAVELENGTH, OURS_N_WAVELENGTHS)]['bci'] * 100:.1f}",
        }
    )

    for n_wavelengths in n_wavelengthses:
        table_data.append(
            {
                "method": f"L={n_wavelengths}",
                "success rate (%)": f"{stats[(OURS_MIN_WAVELENGTH, n_wavelengths)]['mean'] * 100:.1f} ± {stats[(OURS_MIN_WAVELENGTH, n_wavelengths)]['bci'] * 100:.1f}",
            }
        )

    for min_wavelength in min_wavelengths:
        table_data.append(
            {
                "method": f"λ_min={min_wavelength}",
                "success rate (%)": f"{stats[(min_wavelength, OURS_N_WAVELENGTHS)]['mean'] * 100:.1f} ± {stats[(min_wavelength, OURS_N_WAVELENGTHS)]['bci'] * 100:.1f}",
            }
        )

    # Create DataFrame
    df = pd.DataFrame(table_data)

    # Format for clean display
    print(df.to_string(index=False, float_format=lambda x: "{:.3f}".format(x)))

    # Save to markdown
    df.to_markdown(output_dir / f"abl_frequencies.md", index=False)


if __name__ == "__main__":
    main()  # type: ignore[misc]
