from __future__ import annotations

import dataclasses
import logging
from typing import Callable

import torch
import torch.nn as nn
from tensordict import NonTensorData
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.nn import fps, knn

from environments.specs import DataSpecs, EmbedSpec, PointCloudSpec
from transforms.base_transform import KeyMapping, Transform
from utils.nested import cat_nested, pyg_to_nested_tensor

log = logging.getLogger(__name__)


class PointPatchTokenizer(Transform, nn.Module):
    def __init__(
        self,
        specs: DataSpecs,
        embed_dim: int,
        mlp_1: Callable[[int], nn.Module],
        mlp_2: Callable[[int], nn.Module],
        positional_encoder: Callable[[int], nn.Module],
        patch_size: int,
        oversampling_ratio: float,
        fps_random_start: bool = True,
        padding_value: float = 0.0,
        pcd_key: str = "pcd",
    ):
        super().__init__()

        self._input_key = pcd_key
        try:
            self._input_spec = specs.obs[pcd_key]
        except KeyError:
            raise ValueError(
                f"Key {pcd_key} not found in specs. Available keys: {list(specs.obs.keys())}"
            )
        if not isinstance(self._input_spec, PointCloudSpec):
            raise ValueError(
                f"Key {pcd_key} is not a point cloud spec. Found {self._input_spec.type}"
            )

        self.point_dim = 6 if self._input_spec.color else 3
        self.mlp_1 = mlp_1(self.point_dim)
        self.mlp_2 = mlp_2(embed_dim)
        self.pos_encoder = positional_encoder(embed_dim)

        if self.mlp_1.out_features * 2 != self.mlp_2.in_features:
            raise ValueError(
                f"The last layer of mlp_1 (size {self.mlp_1.out_features}) must be half the size of the first layer of mlp_2 (size {self.mlp_2.in_features})"
            )

        self.patch_size = patch_size
        # N_patches * patch_size = N_points * oversampling_ratio
        # fps_sampling_ratio = N_patches / N_points
        # -> fps_sampling_ratio = oversampling_ratio / patch_size
        self.fps_sampling_ratio = oversampling_ratio / patch_size

        self.fps_random_start = fps_random_start
        self.padding_value = padding_value

        # create a modified specs object for the output
        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        if "embed" in obs_specs:
            embed_spec = obs_specs["embed"]
            assert isinstance(embed_spec, EmbedSpec)
            # point clouds are always 1 time step
            if len(embed_spec.shape) == 3 and embed_spec.shape[0] == 1:
                raise ValueError("With point clouds, only one time step is allowed.")

            # the shape includes a none to indicate the jagged dimension
            obs_specs["embed"] = dataclasses.replace(
                embed_spec,
                shape=(None, embed_spec.shape[-1]),
                fixed_shape=False,
            )
            log.debug(
                "Extended obs embedding spec with a variable number of tokens per time step",
            )
        else:
            obs_specs["embed"] = EmbedSpec(shape=(None, embed_dim), fixed_shape=False)
            log.debug(
                f"Created obs embedding spec with a variable number of tokens per time step",
            )
        self._output_specs = specs.replace(obs=obs_specs)

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return [
            KeyMapping(
                in_keys=[("obs", self._input_key), ("obs", "embed")],
                out_keys=[("obs", "embed")],
            )
        ]

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def _call_one(self, nt_data: NonTensorData, obs_embed: Tensor | None) -> Tensor:
        data: Data = nt_data.data  # unpack NonTensorData wrapper around pyg Data object

        assert isinstance(data, Batch)
        pos, batch, color = data.pos, data.batch, data.x
        assert pos is not None
        assert batch is not None

        # pos: (B*N, 3)
        # center_idxs: (B*C)
        center_idxs = fps(
            pos,
            data.batch,
            ratio=self.fps_sampling_ratio,
            random_start=self.fps_random_start,
            batch_size=data.batch_size,
        )

        center_points = pos[center_idxs]  # center_points: (B*C, 3)
        center_batches = batch[center_idxs]  # center_batches: (B*C, 3)

        # find the nearest k points to each center point. these groups of k
        # points become the patches
        # patch_idxs: (B*C*G)
        _, patch_idxs = knn(
            x=pos,
            y=center_points,
            k=self.patch_size,  # G
            batch_x=batch,
            batch_y=center_batches,
            batch_size=data.batch_size,
        )

        patch_pos = pos[patch_idxs]  # patch_pos: (B*C*G, 3)
        patch_pos = patch_pos.view(-1, self.patch_size, 3)  # patch_pos -> (B*C, G, 3)

        # convert to relative coordinates around patch center
        # features: (B*C, G, 3)
        features = patch_pos - center_points.unsqueeze(1)

        if color is not None:
            # concatenate color as an additional feature to the position
            color = color[patch_idxs]  # color -> (B*C*G, 3)
            color = color.view(-1, self.patch_size, 3)  # color -> (B*C, G, 3)
            color = color.to(dtype=features.dtype)
            features = torch.cat([features, color], dim=-1)

        features = self.mlp_1(features)  # features -> (B*C, G, D)

        # max pool over each patch
        aggr_features = torch.max(features, dim=1, keepdim=True).values

        # add the neighborhood max to the original features for each node
        # features -> (B*C, G, 2*D)
        aggr_features = aggr_features.expand(-1, self.patch_size, -1)
        features = torch.cat([aggr_features, features], dim=-1)

        features = self.mlp_2(features)  # features -> (B*C, G, D)

        # max pool over each patch
        features = torch.max(features, dim=1).values  # features -> (B*C, D)

        # add positional encoding
        features += self.pos_encoder(center_points)

        pcd_embed = pyg_to_nested_tensor(features, batch=center_batches)

        if obs_embed is None:
            return pcd_embed

        # concatenate along N dimension of embedding
        # obs_embed: (B, N, D)
        obs_embed = obs_embed.squeeze(1)  # remove time dimension
        obs_embed = cat_nested([obs_embed, pcd_embed], dim=2)
        return obs_embed

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mlp_1={self.mlp_1},(mlp_2={self.mlp_2})"
