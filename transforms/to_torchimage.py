import torch

from environments.datasets.base_dataset import TrajectoryDataset


class ToTorchImage:
    def __init__(self, dataset: TrajectoryDataset) -> None:
        self.rgb_keys = [
            key for key, info in dataset.obs_space.items() if info["type"] == "rgb"
        ]

    def __call__(self, batch):
        default_float_dtype = torch.get_default_dtype()

        obs = batch[0]
        for key in self.rgb_keys:
            obs[key] = (
                torch.movedim(obs[key], -1, -3)  # put it from HWC to CHW format
                .to(dtype=default_float_dtype)  # convert to (some sort of) float
                .div(255)  # rescale to between 0.0 and 1.0
            )

        return batch
