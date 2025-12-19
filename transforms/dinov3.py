from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from tensordict import TensorDict
from torch_geometric.data import Data
from transformers import AutoImageProcessor, AutoModel, pipeline

from environments.specs import DataSpecs
from transforms.base_transform import Transform


class Dinov3FeatureExtractorTransform(Transform):
    """Apply a random translation and rotation to the entire pointcloud
    trajectory during preprocessing, which is then constant throughout
    training.

    Args:
        specs (DataSpecs): The data specifications.
    """

    def __init__(
        self,
        specs: DataSpecs,
        device: str | torch.device = "cuda",
        camera_key: str = "left_cam",
        feature_out_key: str = "dinov3_features",
        dinov3_model: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        dinov3_patch_size: int = 16,
        generate_tracking_points: bool = False,
        tracking_points_key: str = "dinov3_tracking_points",
    ):
        super().__init__()

        self.processor = AutoImageProcessor.from_pretrained(dinov3_model)
        self.model = AutoModel.from_pretrained(
            dinov3_model,
            device_map="auto",
        ).to(device)

        self._specs = specs
        self.device = device

        # General settings
        self.action_seq_len = specs.action_seq_len
        self.camera_key = camera_key
        self.feature_out_key = feature_out_key

        self.dinov3_patch_size = dinov3_patch_size

        self.generate_tracking_points = generate_tracking_points
        self.tracking_points_key = tracking_points_key

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def call_trajectory(self, tensordict: TensorDict) -> TensorDict:
        traj_len = tensordict["action"].shape[0]
        video = (
            tensordict["obs"][self.camera_key]["rgb"]
            .to(self.device)
            .float()
            .permute(0, 3, 1, 2)
            / 255.0
        )
        img_height, img_width = video.shape[2], video.shape[3]
        patches_height = img_height // self.dinov3_patch_size
        patches_width = img_width // self.dinov3_patch_size
        num_patches = patches_height * patches_width

        inputs = self.processor(
            images=video,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
            do_rescale=False,
            do_normalize=True,
        ).to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)

        tensordict["obs"][self.camera_key][self.feature_out_key] = (
            outputs.last_hidden_state[:, :num_patches].reshape(
                traj_len, patches_height, patches_width, -1
            )
        )

        # pca = PCA(n_components=3, whiten=True)
        # starting_img = video[0].permute(1, 2, 0).cpu().numpy()
        # starting_features = (
        #     outputs.last_hidden_state[0].detach().cpu().numpy()[: 14 * 14]
        # )
        # pca.fit(starting_features)
        # transformed_features = pca.transform(starting_features)
        # transformed_features_img = transformed_features.reshape(14, 14, 3)

        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(video[0].permute(1, 2, 0).cpu())
        # ax[1].imshow(transformed_features_img)
        # plt.show()

        # TODO: add start points for tracking
        if self.generate_tracking_points:
            ys = torch.arange(
                self.dinov3_patch_size // 2,
                img_height,
                self.dinov3_patch_size,
                device=self.device,
            )
            xs = torch.arange(
                self.dinov3_patch_size // 2,
                img_width,
                self.dinov3_patch_size,
                device=self.device,
            )

            grid_indices = torch.cartesian_prod(ys, xs)  # (N, 3): (frame, y, x)
            grid_indices = grid_indices.unsqueeze(0).repeat(traj_len, 1, 1)  # (T, N, 2)
            tensordict["obs"][self.camera_key][self.tracking_points_key] = grid_indices

        return tensordict

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        # Do nothing if not called during preprocessing
        return tensordict
