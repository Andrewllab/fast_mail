from __future__ import annotations

import dataclasses
import logging
from typing import Callable

import torch
import torch.nn as nn
from tensordict import NonTensorData
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch, Data
from torch_geometric.nn import fps, knn
from torch_geometric.utils import unbatch

from environments.specs import DataSpecs, EmbedSpec, PointCloudSpec
from transforms.base_transform import KeyMapping, Transform

log = logging.getLogger(__name__)


class PointPatchTokenizer(Transform, nn.Module):
    def __init__(
        self,
        specs: DataSpecs,
        embed_dim: int,
        mlp_1: Callable[[int], nn.Module],
        mlp_2: Callable[[int], nn.Module],
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
            assert len(embed_spec.shape) == 3
            # point clouds are always 1 time step
            assert embed_spec.shape[0] == 1

            # the shape includes a none to indicate the jagged dimension
            obs_specs["embed"] = dataclasses.replace(
                embed_spec,
                shape=(1, None, embed_spec.shape[2]),
                fixed_shape=False,
            )
            log.debug(
                "Extended obs embedding spec with a variable number of tokens per time step",
            )
        else:
            obs_specs["embed"] = EmbedSpec(
                shape=(1, None, embed_dim), fixed_shape=False
            )
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

        # normalize patches around the center points
        patch_pos = patch_pos - center_points.unsqueeze(1)

        if color is not None:
            # concatenate color as an additional feature to the position
            color = color[patch_idxs]  # color -> (B*C*G, 3)
            color = color.view(-1, self.patch_size, 3)  # color -> (B*C, G, 3)
            patch_pos = torch.cat([patch_pos, color], dim=-1)

        features = self.mlp_1(patch_pos)  # features: (B*C, G, D)

        # max pool over each patch
        aggr_features = torch.max(features, dim=1, keepdim=True).values

        # add the neighborhood max to the original features for each node
        aggr_features = aggr_features.expand(-1, self.patch_size, -1)
        features = torch.cat([aggr_features, features], dim=2)

        features = self.mlp_2(features)  # features -> (B*C, G, D)

        # max pool over each patch
        features = torch.max(features, dim=1).values  # features -> (B*C, D)

        # pad each point cloud to the same number of patches
        # pcd_embed: (B, C, D)
        pcd_embed = pad_sequence(
            unbatch(features, center_batches),
            padding_value=self.padding_value,
            batch_first=True,
        )

        # add back a dummy time dimension
        pcd_embed = pcd_embed.unsqueeze(1)  # pcd_embed -> (B, T, C, D)

        # TODO: compute attention mask for the padded sequence

        if obs_embed is not None:
            # concatenate along N dimension of embedding, keeping tokens from the same time step together
            # obs_embed: (B, T, N, D)
            obs_embed = torch.cat([obs_embed, pcd_embed], dim=2)
            return obs_embed
        else:
            return pcd_embed

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mlp_1={self.mlp_1},(mlp_2={self.mlp_2})"


import sys
from typing import Optional

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset


class _PointPatchTokenizer(Transform, MessagePassing):
    def __init__(
        self,
        specs: DataSpecs,
        embed_dim: int,
        mlp_1: Callable[[int], nn.Module],
        mlp_2: Callable[[int], nn.Module],
        patch_size: int,
        oversampling_ratio: float,
        fps_random_start: bool = True,
        padding_value: float = 0.0,
        pcd_key: str = "pcd",
    ):
        super().__init__(aggr="max")

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
        T = self._input_spec.shape[0]
        self.mlp_1 = mlp_1(self.point_dim)
        self.mlp_2 = mlp_2(embed_dim)

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
            assert len(embed_spec.shape) == 3
            assert embed_spec.shape[0] == T

            obs_specs["embed"] = dataclasses.replace(
                embed_spec,
                shape=(T, None, embed_spec.shape[2]),
                fixed_shape=False,
            )
            log.debug(
                "Extended obs embedding spec with a variable number of tokens per time step",
            )
        else:
            obs_specs["embed"] = EmbedSpec(
                shape=(T, None, embed_dim), fixed_shape=False
            )
            log.debug(
                f"Created obs embedding spec with a variable number of tokens per time step",
            )
        self._output_specs = specs.replace(obs=obs_specs)

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.mlp_1)
        reset(self.mlp_2)

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

        center_points_idx = fps(
            pos,
            batch,
            ratio=self.fps_sampling_ratio,
            random_start=self.fps_random_start,
        )
        center_points = pos[center_points_idx]
        batch_y = batch[center_points_idx]

        from_idx, to_idx = knn(
            pos, center_points, self.patch_size, batch_x=batch, batch_y=batch_y
        )
        edges = torch.stack([to_idx, from_idx], dim=0)
        color = (color, color[center_points_idx]) if color is not None else (None, None)
        try:
            x, neighborhoods = self.propagate(edges, pos=(pos, center_points), x=color)
        except RuntimeError as e:
            torch.set_printoptions(threshold=sys.maxsize)
            log.error(
                f"POS SHAPE: {pos.shape}\n BATCH SHAPE: {batch.shape}\n CENTER POINTS SHAPE: {center_points.shape}\n EDGES SHAPE: {edges.shape}\n"
            )
            log.error(f"{type(e).__name__}: {e}")
            raise e

        # pad and reshape into [B, M, E]
        x = pad_sequence(
            unbatch(x, batch_y),
            padding_value=self.padding_value,
            batch_first=True,
        )

        # pad and reshape into [B, G, M, 3]
        neighborhoods = pad_sequence(
            unbatch(
                neighborhoods.reshape(-1, self.patch_size, self.point_dim), batch_y
            ),
            padding_value=self.padding_value,
            batch_first=True,
        )
        # pad and reshape into [B, G, 3]
        center_points = pad_sequence(
            unbatch(center_points, batch_y),
            padding_value=self.padding_value,
            batch_first=True,
        )

        return x, neighborhoods, center_points

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
    ):
        msg, relative_pos = inputs
        return super().aggregate(msg, index, ptr, dim_size), relative_pos

    def message(self, pos_i: Tensor, pos_j: Tensor, x_j: Tensor):
        neighborhood = pos_j - pos_i

        if x_j is not None:
            neighborhood = torch.cat([neighborhood, x_j], dim=1)

        msg = self.mlp_1(neighborhood)
        # reshape into shape [G, M, mlp_1_out_dim]
        msg = msg.reshape(-1, self.patch_size, msg.shape[-1])
        # get max over neighborhood
        msg_max = torch.max(msg, dim=1, keepdim=True)[0]
        # add the neighborhood max to the original msg for each node
        msg = torch.cat([msg_max.expand(-1, self.patch_size, -1), msg], dim=2)
        msg = self.mlp_2(msg.reshape(-1, msg.shape[-1]))

        return msg, neighborhood

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mlp_1={self.mlp_1},(mlp_2={self.mlp_2})"
