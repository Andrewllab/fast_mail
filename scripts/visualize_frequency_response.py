import logging
import re
from pathlib import Path
from typing import Callable

import hydra
import matplotlib.pyplot as plt
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf

# enables importing local modules regardless of where the script is run
rootutils.setup_root(__file__, indicator=".isort.cfg", pythonpath=True)

from torch.optim.swa_utils import AveragedModel

from agents.base_agent import BaseAgent
from loggers.wandb import resolve_checkpoint
from utils.conf import (
    delete_keys_recursively,
    patch_legacy_targets,
    patch_load_from_checkpoint,
    setup_resolvers,
)
from utils.frequency_response import compute_model_fft
from utils.logging import configure_logging, log_exception_and_finish_wandb
from utils.torch_conf import configure_torch

log = logging.getLogger(__name__)


def get_pointpatch_tokenizer_mlps(agent: BaseAgent) -> dict[str, Callable]:
    obs_encoder = agent.obs_encoder
    if isinstance(obs_encoder, AveragedModel):
        obs_encoder = obs_encoder.module
    pcd_tokenizer = obs_encoder["11_tokenizer"]
    point_encoders = pcd_tokenizer.point_encoders
    return point_encoders


def get_model_ffts(
    point_encoders: dict[str, Callable],
    direction: dict[str, torch.Tensor],
    cfg: DictConfig,
    device=None,
) -> dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

    direction = {
        key: torch.as_tensor(val, dtype=torch.float32, device=device)
        for key, val in direction.items()
    }

    defaults_cfg = cfg.get("defaults", {}) or {}

    model_names = cfg.model_names
    model_ffts: dict[
        tuple[str, str], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = {}

    for name in model_names:
        if name not in point_encoders:
            log.warning("Point tokenizer missing expected module %s", name)
            continue

        module_cfg = dict(defaults_cfg)
        module_cfg.update(cfg.get(name, {}) or {})

        point = module_cfg["point"]
        min_wavelength = module_cfg["min_wavelength"]
        n_steps = module_cfg["n_steps"]
        n_bins = module_cfg["n_bins"]

        point = torch.as_tensor(point, dtype=torch.float32, device=device)

        # axis-aligned FFT (x-axis)
        freqs_axis, freq_bins_axis, fft_axis = compute_model_fft(
            model=point_encoders[name],
            point=point,
            direction=direction["axis_aligned"],
            min_wavelength=min_wavelength,
            n_steps=n_steps,
            n_bins=n_bins,
        )

        # omnidirectional FFT averaged later across random directions
        freqs_omni, freq_bins_omni, fft_omni = compute_model_fft(
            model=point_encoders[name],
            point=point,
            direction=direction["omnidirectional"],
            min_wavelength=min_wavelength,
            n_steps=n_steps,
            n_bins=n_bins,
        )

        model_ffts[(name, "axis_aligned")] = (
            freqs_axis.cpu(),
            freq_bins_axis.cpu(),
            fft_axis.cpu(),
        )
        model_ffts[(name, "omnidirectional")] = (
            freqs_omni.cpu(),
            freq_bins_omni.cpu(),
            fft_omni.cpu(),
        )

    return model_ffts


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="visualize_frequency_response",
)
@log_exception_and_finish_wandb
def test(cfg: DictConfig) -> None:
    # resolve the entire config to catch any errors early
    OmegaConf.resolve(cfg)

    configure_logging(cfg.python_logging)

    # configure torch, e.g. set_float32_matmul_precision
    configure_torch(cfg.get("torch"))

    do_show = cfg.get("do_show", True)
    do_tikz = cfg.get("do_tikz", False)
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    agent_defaults = cfg.get("agent", {})
    device = "cuda" if cfg.trainer.accelerator == "gpu" else "cpu"
    checkpoint_defaults = cfg.get("checkpoint_defaults", {})
    checkpoints = cfg.get("checkpoints", [])
    n_directions = cfg.frequency_response.get("n_directions", 256)

    direction = {
        # use x-axis only
        "axis_aligned": torch.tensor([1, 0, 0], dtype=torch.float32, device=device),
        # # create a set of random directions that we can use for all models
        # # for a fair comparison
        "omnidirectional": torch.randn(
            (n_directions, 3), dtype=torch.float32, device=device
        ),
    }

    ffts = {}
    for checkpoint_cfg in checkpoints:
        # merge in any default checkpoint config settings
        checkpoint_cfg = {**checkpoint_defaults, **checkpoint_cfg}

        checkpoint = resolve_checkpoint(checkpoint_cfg)
        # load agent config and specs from checkpoint
        run, train_cfg, checkpoint_paths = checkpoint
        if isinstance(run, Path):
            run_name = run.name
        else:
            # wandb Api Run object
            run_name = run.id

        # use the first checkpoint to initialize the agent, if multiple are given
        epoch, checkpoint_path = checkpoint_paths[0]
        log.info(f"Loading checkpoint after epoch {epoch} of run {run_name}...")

        # merge the agent and data configs, with the current config taking precedence
        agent_cfg = OmegaConf.merge(train_cfg.agent, agent_defaults)

        # replace legacy _target_ with updated ones
        agent_cfg = patch_legacy_targets(agent_cfg)

        # modify the _target_ to point to the module's load_from_checkpoint method
        agent_cfg = patch_load_from_checkpoint(agent_cfg, checkpoint_path)

        # instantiate agent
        log.debug("Instantiating agent...")
        # recursively delete these fields in config dictionary
        # we want these to be saved to WandB but we don't want them for instantiation
        delete_keys_recursively(agent_cfg, ["name"])
        agent: BaseAgent = hydra.utils.instantiate(agent_cfg)
        agent = agent.to(device)
        agent.checkpoint_metadata = {"run_name": run_name, "epoch": epoch}

        log.info(f"Testing run {run_name} at epoch {epoch}...")
        # get relevant tokenizer from agent
        point_encoders = get_pointpatch_tokenizer_mlps(agent)
        device = agent.device
        model_ffts = get_model_ffts(
            point_encoders=point_encoders,
            direction=direction,
            cfg=cfg.frequency_response,
            device=device,
        )
        model_ffts = {
            (run_name, epoch, model_name, mode): value
            for (model_name, mode), value in model_ffts.items()
        }
        ffts.update(model_ffts)

        # if multiple checkpoints are given, test them all
        for epoch, checkpoint_path in checkpoint_paths[1:]:
            log.info(f"Loading checkpoint at epoch {epoch} of run {run_name}...")
            # this is adapted from LightningModule.load_from_checkpoint but we don't
            # want to re-instantiate the model
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            agent.on_load_checkpoint(checkpoint)
            agent.load_state_dict(checkpoint["state_dict"], strict=agent.strict_loading)
            agent.checkpoint_metadata = {"run_name": run_name, "epoch": epoch}

            log.info(f"Testing run {run_name} at epoch {epoch}...")
            # get relevant tokenizer from agent
            point_encoders = get_pointpatch_tokenizer_mlps(agent)
            device = agent.device
            model_ffts = get_model_ffts(
                point_encoders=point_encoders,
                direction=direction,
                cfg=cfg.frequency_response,
                device=device,
            )
            model_ffts = {
                (run_name, epoch, model_name, mode): value
                for (model_name, mode), value in model_ffts.items()
            }
            ffts.update(model_ffts)

    log.info("Creating plots...")

    fig, axs = plt.subplots(3, 2, figsize=(14, 8), sharey="row")

    plot_layout = [
        (
            "spatial_encoder",
            "axis_aligned",
            (0, 0),
        ),
        (
            "spatial_encoder",
            "omnidirectional",
            (0, 1),
        ),
        (
            "mlp1",
            "axis_aligned",
            (1, 0),
        ),
        (
            "mlp1",
            "omnidirectional",
            (1, 1),
        ),
        (
            "patch_pos_encoder",
            "axis_aligned",
            (2, 0),
        ),
        (
            "patch_pos_encoder",
            "omnidirectional",
            (2, 1),
        ),
    ]

    # create a consistent color mapping across all subplots
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors_and_labels = {
        "hqns9h68": (colors[0], "ours"),
        "490ob783": (colors[1], "no FFs"),
        "3kegwtez": (colors[2], "logspace + SPE"),
        "pzz1m3s3": (colors[3], "RFF"),
        "qxjq2io7": (colors[4], "RFF + learned"),
        "btvlnpwk": (colors[5], "RFF + SPE"),
        "hnk6118s": (colors[6], "RFF + Cartesian"),
    }

    column_headings = ["X-Axis", "Omnidirectional"]
    row_headings = [
        "Fourier Features",
        "Local Patch Encoder",
        "Patch Position\nEncoder",
    ]

    def reduce_fft(fft_tensor: torch.Tensor) -> torch.Tensor:
        if fft_tensor.ndim > 2:
            fft_tensor = fft_tensor.mean(dim=0)
        assert fft_tensor.ndim == 2
        return fft_tensor.mean(dim=-1)

    # loop over the declared subplot layout and plot the corresponding FFTs
    for module_name, mode, (row, col) in plot_layout:
        ax = axs[row][col]
        model_ffts = {
            key: value
            for key, value in ffts.items()
            if key[2] == module_name and key[3] == mode
        }

        for (run_name, epoch, _, _), (freqs, freq_bins, fft) in model_ffts.items():
            amplitude = reduce_fft(fft)
            color, label = colors_and_labels[run_name]
            ax.plot(freqs, amplitude, label=label, color=color)

        ax.set_xscale("log")

        # show every second label to reduce clutter
        freq_labels = [f"{(1/freq):.3f}" for freq in freqs[::2]]
        ax.set_xticks(freqs[::2], freq_labels, minor=False, rotation=45, ha="center")
        # add the remaining ticks as minor ticks without labels
        ax.set_xticks(freqs, [], minor=True)

    # add column headings
    for col, col_title in enumerate(column_headings):
        axs[0][col].set_title(col_title)
        # axs[-1][col].set_xlabel("Wavelength (m)")

    # add row headings
    for row, row_title in enumerate(row_headings):
        axs[row][0].set_ylabel(row_title, rotation=90, labelpad=40, va="center")

    # Add a single common label for x and y axes
    fig.supylabel("FFT Amplitude", fontsize=12)
    fig.supxlabel("Wavelength (m)", fontsize=12)

    # create a dictionary of labels to handles to remove duplicates
    unique_legend = {
        # swap to label: handle from (handle, lavel) returned by get_legend_handles_labels
        label: handle
        # loop over all axes to collect all legend entries
        for row in axs
        for ax in row
        # loop over all legend entries in the axis
        for handle, label in zip(*ax.get_legend_handles_labels())
    }

    # reorder legend according to colors_and_labels
    unique_legend = {
        label: unique_legend[label]
        for color, label in colors_and_labels.values()
        if label in unique_legend
    }

    fig.tight_layout()

    if do_tikz:
        import matplot2tikz

        tikz_code = matplot2tikz.get_tikz_code(
            axis_width=r"0.5\textwidth", axis_height=r"0.25\textwidth", textsize=10
        )

        # escape unescaped %} sequences that break LaTeX compilation
        tikz_code = re.sub(r"(?<!\\)%\}", r"\%}", tikz_code)

        filename = "frequency_response.tex"
        with open(output_dir / filename, "w") as f:
            f.write(tikz_code)
        log.info(f"Saved tex code to {output_dir / filename}")

    if do_show:
        fig.legend(unique_legend.values(), unique_legend.keys(), loc="upper right")

        filename = "frequency_response.png"
        fig.savefig(str(output_dir / filename))
        log.info(f"Saved plot to {output_dir / filename}")

        plt.show()

    log.info("Done.")


if __name__ == "__main__":
    setup_resolvers()
    test()
