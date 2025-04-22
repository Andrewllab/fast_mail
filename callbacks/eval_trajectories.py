import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks.callback import Callback
from typing_extensions import override

log = logging.getLogger(__name__)

DIM_NAMES = ["x", "y", "z", "quat_w", "quat_x", "quat_y", "quat_z", "gripper"]
FOLDER_NAME = "trajectory_plots"


class EvaluatePredictedTrajectories(Callback):
    def __init__(self, action_horizons: Sequence[int], plot: bool = False) -> None:
        self.action_horizons = list(action_horizons)
        self.plot = plot

        self.target_actions = []
        self.predicted_actions = []

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        self.stage = stage
        self.target_actions.clear()
        self.predicted_actions.clear()

    @override
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.target_actions.append(batch["action"].cpu())
        self.predicted_actions.append(outputs["actions"].cpu())

    on_test_batch_end = on_validation_batch_end

    @override
    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:

        target_actions = torch.cat(self.target_actions, dim=0)
        predicted_actions = torch.cat(self.predicted_actions, dim=0)

        action_seq_len = target_actions.shape[1]
        traj_lengths = trainer.datamodule.specs.lengths
        samples_per_trajectory = torch.tensor(traj_lengths) - (action_seq_len - 1)
        assert samples_per_trajectory.sum() == target_actions.shape[0]
        cumulative_samples = F.pad(torch.cumsum(samples_per_trajectory, dim=0), (1, 0))

        log_dir = Path(trainer.log_dir) / FOLDER_NAME
        if self.plot:
            log_dir.mkdir(parents=True, exist_ok=True)

        for action_horizon in self.action_horizons:
            losses = []
            for i in range(len(traj_lengths)):
                start = cumulative_samples[i]
                end = cumulative_samples[i + 1]

                target_action = target_actions[start:end]
                predicted_action = predicted_actions[start:end]

                target_action = torch.flatten(
                    target_action[::action_horizon, :action_horizon],
                    end_dim=1,
                )
                predicted_action = torch.flatten(
                    predicted_action[::action_horizon, :action_horizon],
                    end_dim=1,
                )

                if self.plot:
                    import matplotlib.pyplot as plt

                    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
                    axs = axs.flatten()

                    for dim in range(target_action.shape[-1]):
                        ax = axs[dim]
                        ax.plot(target_action[..., dim].numpy(), label="Target")
                        ax.plot(predicted_action[..., dim].numpy(), label="Predicted")
                        ax.set_title(DIM_NAMES[dim])
                        ax.set_xlabel("Time")
                        ax.set_ylabel("Value")
                        ax.legend()

                    fig.suptitle(
                        f"Action horizon={action_horizon}, Trajectory {i + 1}",
                        fontsize=16,
                    )
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                    filename = f"traj_{i + 1}_horizon_{action_horizon}.png"
                    plt.savefig(log_dir / filename)
                    plt.close(fig)  # Close the figure to free memory
                    log.debug(f"Saved plot to {filename}")

                losses.append(F.mse_loss(predicted_action, target_action))
            loss = torch.mean(torch.stack(losses))
            pl_module.log(
                f"{self.stage}/trajectory_mse_(action_horizon={action_horizon})",
                loss,
                on_epoch=True,
            )

        self.target_actions.clear()
        self.predicted_actions.clear()

    on_test_epoch_end = on_validation_epoch_end
