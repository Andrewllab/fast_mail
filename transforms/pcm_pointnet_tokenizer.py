from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn as nn
from tensordict import NonTensorData
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.nn.aggr import MaxAggregation

from environments.specs import DataSpecs, EmbedSpec, PointCloudSpec
from transforms.base_transform import KeyMapping, Transform
from utils.nested import cat_nested
from utils.pyg import batch2ptr, fps, knn

log = logging.getLogger(__name__)


class PCMPointNetTokenizer(Transform, nn.Module):
    r"""Tokenizer for point clouds as described in the paper `Point Cloud Matters:
    Rethinking the Impact of Different Observation Spaces on Robot Learning
    <https://arxiv.org/abs/2402.02500>`__ .

    This tokenizer implicitly used the "PointNet" backbone from the paper.

    Reference: https://github.com/HaoyiZhu/PointCloudMatters/blob/main/src/models/components/diffusion_policy/vision/pcd_obs_encoder.py
    """

    def __init__(
        self,
        specs: DataSpecs,
        embed_dim: int,
        backbone: Callable[[int], nn.Linear],
        n_patches: int,
        patch_size: int,
        point_mlp: Callable[[int], nn.Linear],
        patch_mlp: Callable[[int], nn.Linear],
        head_mlp: Callable[[int, int], nn.Linear],
        norm: Callable[[int | tuple[int, ...]], nn.Module] | str | None = None,
        activation: Callable[[], nn.Module] | str | None = nn.ReLU,
        spatial_encoder: Callable[[int], nn.Linear] | None = None,
        fps_random_start: bool = True,
        pre_sample: bool = False,
        pcd_key: str = "pcd",
        token_pos_encoder: Callable[[int, int], nn.Module] | None = None,
    ):
        super().__init__()

        self._input_key = pcd_key
        try:
            self._input_spec: PointCloudSpec = specs.obs[pcd_key]
        except KeyError:
            raise ValueError(
                f"Key '{pcd_key}' not found in specs. Available keys: {list(specs.obs.keys())}"
            )
        if not isinstance(self._input_spec, PointCloudSpec):
            raise ValueError(
                f"Key '{pcd_key}' is not a PointCloudSpec. Found {self._input_spec.type}"
            )

        spatial_dim = 3
        if spatial_encoder is not None:
            spatial_encoder = spatial_encoder(spatial_dim)
            spatial_dim = spatial_encoder.out_features
        self.spatial_encoder = spatial_encoder

        point_dim = spatial_dim
        if self._input_spec.color:
            point_dim += 3

        # transform the point features (absolute position and maybe color)
        self.backbone = backbone(point_dim, norm=norm, activation=activation)
        point_dim = self.backbone.out_features

        # increment point_dim since we concatenate relative point positions
        # onto the features (maybe after spatial encoding)
        point_dim += spatial_dim

        # transform the point features after concatenation with relative position
        self.point_mlp = point_mlp(
            point_dim,
            norm=norm,
            activation=activation,
            # apply norm and activation after final layer
            plain_last=False,
        )
        point_dim = self.point_mlp.out_features

        # transform the embeddings of each patch
        self.patch_mlp = patch_mlp(
            point_dim,
            norm=norm,
            activation=activation,
            # apply norm and activation after final layer
            plain_last=False,
        )
        feature_dim = self.patch_mlp.out_features

        # aggregate over all patches
        self.max_aggr = MaxAggregation()

        # transform the embedding of the point cloud
        self.head_mlp = head_mlp(
            feature_dim,
            embed_dim,
            norm=norm,
            # apply norm but not activation after final layer
            plain_last=False,
            activation=None,
        )

        self.pre_sample = pre_sample
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.fps_random_start = fps_random_start

        # create encoder for token position
        if token_pos_encoder is None:
            log.warning("No token position encoder provided. Using nn.Embedding.")
            token_pos_encoder = nn.Embedding
        self.token_pos_encoder = token_pos_encoder(1, embed_dim)

        new_spec = EmbedSpec(embed_dim=embed_dim, n_tokens=1)
        obs_specs = dict(specs.obs)
        if "embed" in obs_specs:
            embed_spec = obs_specs["embed"]
            assert isinstance(embed_spec, EmbedSpec)
            new_spec = embed_spec.concat(new_spec)
        obs_specs["embed"] = new_spec
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

    def fps_and_knn(
        self, pos: Tensor, ptr: Tensor, batch: Tensor, features: Tensor
    ) -> tuple[Tensor, Tensor]:
        # pos: (B*N, 3)
        # center_idxs: (B*C)
        center_idxs = fps(
            pos, ptr=ptr, n_points=self.n_patches, random_start=self.fps_random_start
        )

        center_pos = pos[center_idxs]  # center_pos: (B*C, 3)
        center_batch = batch[center_idxs]  # center_batch: (B*C, 3)
        n_patches = center_idxs.size(0)

        # find the nearest k points to each center point. these groups of k
        # points become the patches
        # patch_idxs: (B*C*G)
        _, patch_idxs = knn(
            x=pos,
            y=center_pos,
            k=self.patch_size,
            ptr_x=ptr,
            batch_y=center_batch,
            batch_size=ptr.size(0) - 1,
            pad_too_small=True,
        )

        patch_pos = pos[patch_idxs]  # patch_pos: (B*C*G, 3)
        # patch_pos -> (B*C, G, 3)
        patch_pos = patch_pos.unflatten(dim=0, sizes=(n_patches, self.patch_size))

        # convert to relative coordinates around patch center
        patch_pos = patch_pos - center_pos.unsqueeze(1)

        if self.spatial_encoder is not None:
            # patch_pos -> (B*N, D)
            patch_pos = self.spatial_encoder(patch_pos)

        # concatenate the relative coordinates of each point with the other
        # point features
        features = features[patch_idxs]  # features -> (B*C*G, D)
        # features -> (B*C, G, D)
        features = features.unflatten(dim=0, sizes=(n_patches, self.patch_size))
        features = torch.cat([features, patch_pos], dim=-1)

        # transform the point features after concatenation with relative position
        features = self.point_mlp(features)

        # max pool over each patch
        # features -> (B*C, D)
        features = torch.max(features, dim=-2).values

        return features, center_batch

    def _call_one(self, nt_data: NonTensorData, obs_embed: Tensor | None) -> Tensor:
        data: Data = nt_data.data  # unpack NonTensorData wrapper around pyg Data object

        assert isinstance(data, Data)
        assert isinstance(data, Batch)

        pos, color = data.pos, data.x
        batch, ptr, batch_size = data.batch, data.ptr, data.batch_size
        assert pos is not None
        assert batch is not None
        assert ptr is not None
        assert batch_size is not None

        # verify that ptr is up to date
        if data.ptr[-1] != batch.shape[0]:
            ptr = batch2ptr(batch, batch_size=batch_size)

        features = pos

        if self.spatial_encoder is not None:
            # features -> (B*N, D)
            features = self.spatial_encoder(features)

        if color is not None:
            features = torch.cat([features, color], dim=-1)

        if self.pre_sample:
            raise NotImplementedError
        else:
            # 1. use pointnet on full (voxelized) pc
            # features: (B*N, D)
            features = self.backbone(features)

            # 2.,3.,4. fps + KNN on sampled points + group with features
            # features: (B*C, D)
            features, batch = self.fps_and_knn(pos, ptr, batch, features)
            ptr = batch2ptr(batch, batch_size=batch_size)

        # features: (B*C, D)
        features = self.patch_mlp(features)

        # max pool over all patches in each batch element
        features = self.max_aggr(features, batch, ptr=ptr, dim_size=batch_size, dim=-2)

        # features -> (B, D)
        features = self.head_mlp(features)

        # features -> (B, 1, D)
        features = features.unsqueeze(dim=1)

        # add encoding of the token position to the token
        token_indices = torch.arange(1, dtype=torch.long, device=features.device)
        token_pos_embed = self.token_pos_encoder(token_indices)
        features += token_pos_embed

        if obs_embed is None:
            return features

        # concatenate along N dimension of embedding
        # obs_embed: (B, N, D)
        return cat_nested([obs_embed, features], dim=-2)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(n_patches={self.n_patches}, "
            f"patch_size={self.patch_size}, embed_dim={self._output_specs.obs['embed'].embed_dim})"
        )
