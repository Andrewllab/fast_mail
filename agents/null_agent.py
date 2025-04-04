import torch

from agents.base_agent import BaseAgent
from transforms.base_transform import TransformPartialsDict


class NullAgent(BaseAgent):
    """The name of the class makes it sound super cool but actually this agent just does nothing."""

    def __init__(
        self,
        specs,
        obs_encoder: TransformPartialsDict,
    ):
        super().__init__(
            model=lambda specs: None,
            obs_encoder=obs_encoder,
            optimizer=None,
            lr_scheduler=None,
            scaler=lambda specs: None,
            goal_encoder=None,
            specs=specs,
        )

    def configure_optimizers(self):
        raise NotImplementedError(
            "Null agent is not suitable for training. Use a different agent."
        )

    def test_step(self, batch, batch_idx):
        # just run any transforms on the batch
        batch = self.obs_encoder(batch)

        actions = torch.zeros(
            batch.batch_size + self.specs.action.shape, device=batch.device
        )
        return actions

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> torch.Tensor:
        # just run any transforms on the batch
        batch = self.obs_encoder(batch)

        actions = torch.zeros(
            batch.batch_size + self.specs.action.shape, device=batch.device
        )
        return actions
